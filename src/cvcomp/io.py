import base64
import json
import multiprocessing
import os
import queue
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Type, Any, Callable

import cv2
import numpy as np
import redis

from cvcomp.models.base import CVModel


class Detector(ABC):

    def __init__(self, model_name: str):
        self._model_name = model_name

    def get_name(self) -> str:
        return self._model_name

    @abstractmethod
    def is_busy(self) -> bool:
        """Returns True if the detector is busy"""
        pass

    @abstractmethod
    def ingest(self, frame: np.ndarray) -> bool:
        """Send a frame to the transport"""
        pass

    @abstractmethod
    def get_result(self, block: Optional[bool] = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Get the result from the transport"""
        pass

    @abstractmethod
    def stop(self):
        pass


class RedisFrameDispatcher(Detector):

    def __init__(
            self,
            model_name: str,
            redis_url: str,
            capacity_channel="capacity",
            result_channel="results",
            beacon_channel="beacon",
    ):
        super().__init__(model_name)
        self._available_workers = []
        self._beacon_channel = beacon_channel

        self._redis = redis.from_url(redis_url)
        self._stop_event = threading.Event()
        # self._frames_out = queue.Queue()

        self._capacity_channel = capacity_channel
        self._capacity_thread = threading.Thread(target=self._run_capacity_thread)
        self._capacity_thread.start()

        self._result_channel = result_channel
        self._result_q = queue.Queue()
        self._results_thread = threading.Thread(target=self._run_results_thread)
        self._results_thread.start()

        self._remote_worker_lock = threading.Lock()

        # self._publisher_thread = threading.Thread(target=self._run_publisher_thread)
        # self._publisher_thread.start()

        self._beacon_thread = threading.Thread(target=self._run_beacon_thread)
        self._beacon_thread.start()

    def ingest(self, frame: np.ndarray):
        with self._remote_worker_lock:
            if len(self._available_workers) == 0:
                raise RuntimeError("No available workers")
            topic = self._available_workers.pop()
        frame = _frame_encode(frame)
        self._redis.publish(topic, frame)

    def get_result(self, block: Optional[bool] = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self._result_q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def is_busy(self) -> bool:
        with self._remote_worker_lock:
            return len(self._available_workers) == 0

    def _run_capacity_thread(self):
        print("starting capacity thread")
        chan = self._redis.pubsub()
        chan.subscribe(self._capacity_channel)
        while not self._stop_event.is_set():
            msg = chan.get_message(ignore_subscribe_messages=True, timeout=1)
            if msg is None:
                continue
            data = msg.get("data", b"").decode("utf-8")
            data = json.loads(data)
            print(f"frame dispatcher: received capacity {data}")
            if "model" not in data or "topic" not in data or "capacity" not in data:
                # TODO: log error
                continue
            model_name = data["model"]
            topic = data["topic"]
            # TODO: capacity must be a sane positive number. just trust this for now
            capacity = int(data["capacity"])

            if model_name == self._model_name:
                with self._remote_worker_lock:
                    self._available_workers.extend([topic] * capacity)
                    print(f"frame dispatcher: available workers {self._available_workers}")

        chan.unsubscribe(self._capacity_channel)

    def _run_results_thread(self):
        """Listen for results from remote workers"""
        print("starting results thread")
        chan = self._redis.pubsub()
        chan.subscribe(self._result_channel)
        while not self._stop_event.is_set():
            msg = chan.get_message(ignore_subscribe_messages=True, timeout=1)
            if msg is None:
                continue
            print(f"frame dispatcher: received result")
            frame = _frame_decode(msg["data"])
            self._result_q.put(frame)
            print("frame dispatcher: result put in queue")
        chan.unsubscribe(self._result_channel)

    def _run_beacon_thread(self):
        """Listen for beacons from remote workers and give them the binding info"""
        print("starting beacon thread")
        chan = self._redis.pubsub()
        chan.subscribe(self._beacon_channel)
        while not self._stop_event.is_set():
            msg = chan.get_message(ignore_subscribe_messages=True, timeout=1)
            if msg is None:
                continue
            print(f"frame dispatcher: received beacon {msg}")
            data = msg.get("data", b"").decode("utf-8")
            data = json.loads(data)
            if "model" not in data or "respond_to" not in data:
                # TODO: log error
                continue
            model_name = data["model"]
            print(f"frame dispatcher: received beacon for model name {model_name}")
            if model_name != self._model_name:
                continue
            respond_to = data["respond_to"]
            print(f"frame dispatcher: responding to beacon {respond_to}")
            self._redis.publish(
                respond_to,
                json.dumps(
                    {
                        "model": self._model_name,
                        "capacity_channel": self._capacity_channel,
                        "result_channel": self._result_channel,
                    }
                )
            )

    def stop(self):
        # TODO: wait for threads to stop
        self._stop_event.set()


class RedisFrameConsumer:

    def __init__(
            self,
            model_name: str,
            redis_url: str,
            model_factory: Optional[Callable[[], CVModel]],
            capacity: int = 1,
            beacon_channel="beacon",
    ):
        self._model_name = model_name
        self._beacon_channel = beacon_channel

        self._redis = redis.from_url(redis_url)

        my_id = str(uuid.uuid4())
        self._beacon_response_channel = f"beacon-response:{my_id}"
        self._frames_in_channel = f"frames-in:{my_id}"

        self._capacity_channel = None
        self._result_channel = None
        self._stop_event = threading.Event()

        self._configuration_request_thread = threading.Thread(target=self._run_configuration_request_thread)
        self._configuration_request_thread.start()

        self._frame_ingester: Optional[threading.Thread] = None

        self._capacity = capacity
        self._model = MultiprocessDetector(
            model_name, model_factory,
            capacity=capacity,
        )

    def _run_configuration_request_thread(self):
        print("frame consumer: starting beacon thread")
        chan = self._redis.pubsub()
        chan.subscribe(self._beacon_response_channel)
        while not self._stop_event.is_set():
            print(f"frame consumer: publishing beacon on {self._beacon_channel}")
            self._redis.publish(
                self._beacon_channel,
                json.dumps(
                    {
                        "model": self._model_name,
                        "respond_to": self._beacon_response_channel,
                    }
                )
            )
            msg = chan.get_message(ignore_subscribe_messages=True, timeout=1)
            if msg is None:
                continue
            data = msg.get("data", b"").decode("utf-8")
            data = json.loads(data)
            print(f"frame consumer: got beacon response: {data}")
            if "model" not in data or "capacity_channel" not in data or "result_channel" not in data:
                # TODO: log error
                continue
            model_name = data["model"]
            if model_name != self._model_name:
                continue
            self._capacity_channel = data["capacity_channel"]
            self._result_channel = data["result_channel"]
            chan.unsubscribe(self._beacon_channel)
            self._bind_and_accept_work()
            return

    def _bind_and_accept_work(self):
        if self._frame_ingester is not None:
            raise RuntimeError("Worker already running")
        self._frame_ingester = threading.Thread(target=self._run_frame_ingestion)
        self._frame_ingester.start()

        self._result_dequeuer = threading.Thread(target=self._run_result_deque)
        self._result_dequeuer.start()

        self._frame_ingester.join()

    def _run_frame_ingestion(self):
        chan = self._redis.pubsub()
        print(f"frame consumer: subscribing to frames channel {self._frames_in_channel}")
        chan.subscribe(self._frames_in_channel)
        self._announce_capacity(self._capacity)
        while not self._stop_event.is_set():
            print("frame consumer: waiting for frame to process")
            msg = chan.get_message(ignore_subscribe_messages=True, timeout=1)
            if msg is None:
                continue
            data = _frame_decode(msg["data"])
            self._model.ingest(data)
        chan.unsubscribe(self._frames_in_channel)

    def _run_result_deque(self):
        while not self._stop_event.is_set():
            result = self._model.get_result(block=True, timeout=1)
            if result is not None:
                print("frame consumer: publishing result")
                self._redis.publish(self._result_channel, _frame_encode(result))
                self._announce_capacity(1)

    def _announce_capacity(self, capacity: int):
        print(f"frame consumer: announcing capacity {capacity}")
        self._redis.publish(
            self._capacity_channel,
            json.dumps({"model": self._model_name, "topic": self._frames_in_channel, "capacity": capacity})
        )


def _frame_encode(frame: np.ndarray) -> str:
    offset = 500
    # TODO: use a better way to serialize metadata
    metadata = f"{frame.shape}#{frame.dtype}".encode("utf-8").ljust(offset, b" ")
    frame = metadata + frame.tobytes()
    encoded = base64.b64encode(frame).decode("utf-8")
    return encoded


def _frame_decode(frame: str) -> np.ndarray:
    offset = 500
    frame = base64.b64decode(frame)
    metadata = frame[:offset]
    shape, dtype = metadata.decode("utf-8").strip().split("#")
    shape = tuple(int(x) for x in shape[1:-1].split(", "))
    frame = np.frombuffer(frame, dtype=dtype, offset=offset).reshape(shape)
    return frame


def _process_frames(
        queue_in: multiprocessing.Queue,
        queue_out: multiprocessing.Queue,
        model_factory: Callable[[], Detector]
):
    pid = os.getpid()
    print(f"worker {pid} started", flush=True)
    model = model_factory()

    while True:
        # get PID of current process
        frame = queue_in.get()
        if frame is None:
            print(f"worker {pid}: exiting", flush=True)
            queue_out.close()
            # cancel_join_thread is needed to avoid a deadlock when this process exits
            # if we don't do this, the main process won't be able to clean the out_queue
            # and will hang on join
            queue_out.cancel_join_thread()
            queue_in.cancel_join_thread()
            return
        result = model.detect(frame)
        queue_out.put(result)


class MultiprocessDetector(Detector):
    """A frame transport that uses a multiprocessing queue to send frames"""

    def __init__(
            self,
            model_name: str,
            model_factory: Optional[Callable[[], CVModel]] = None,
            capacity: int = 1,
    ):
        super().__init__(model_name)
        self._model_name = model_name
        self._q_in = multiprocessing.Queue()
        self._q_out = multiprocessing.Queue()
        self._max_workers = capacity
        self._capacity = capacity

        if model_factory is not None:
            self._workers = [
                multiprocessing.Process(
                    target=_process_frames,
                    args=(self._q_in, self._q_out, model_factory),
                    daemon=False,
                ) for _ in range(self._max_workers)
            ]
        for worker in self._workers:
            worker.start()
        print(f"multiprocess detector: started {len(self._workers)} workers")

    def _set_capacity(self, capacity: int):
        # TODO: this is probably not thread safe
        print("multiprocess detector: setting capacity", capacity)
        self._capacity = capacity

    def ingest(self, frame: np.ndarray) -> int:
        if self.is_busy():
            raise ValueError("Detector is busy")
        print("multiprocess detector: ingesting frame")
        self._set_capacity(self._capacity - 1)
        self._q_in.put(frame, block=True)
        print("multiprocess detector: frame ingested")
        return self._capacity

    def get_result(self, block: Optional[bool] = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            result = self._q_out.get(block=block, timeout=timeout)
            if result is not None:
                self._set_capacity(self._capacity + 1)
            return result
        except queue.Empty:
            return None

    def is_busy(self) -> bool:
        return self._capacity == 0

    def stop(self):
        for _ in range(self._max_workers):
            self._q_in.put(None)
        # To prevent a deadlock, it is necessary to clear the output queue
        # before joining the child process.
        # Refer to the warnings in the Python documentation:
        # https://docs.python.org/3/library/multiprocessing.html?highlight=queue#pipes-and-queues
        _drain_queue(self._q_out)
        for w in self._workers:
            w.join()


def _drain_queue(q: multiprocessing.Queue):
    while True:
        try:
            q.get(block=False)
        except queue.Empty:
            break


class ModelMuxer(Detector):

    def __init__(self, detectors: list[Detector], model_name: str):
        super().__init__(model_name)
        self._detectors = detectors
        self._results = {i: None for i in range(len(self._detectors))}

    def is_busy(self) -> bool:
        return any(d.is_busy() for d in self._detectors)

    def ingest(self, frame: np.ndarray) -> bool:
        for detector in self._detectors:
            detector.ingest(frame)
        return True

    def get_result(self, block: Optional[bool] = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        for k, v in self._results.items():
            if v is not None:
                continue
            self._results[k] = self._detectors[k].get_result(block=block, timeout=timeout)

        if all([v is not None for v in self._results.values()]):
            frame = cv2.hconcat(
                [self._results[i] for i in self._results.keys()]
            )
            self._results = {i: None for i in self._results.keys()}
            return frame

    def stop(self):
        for detector in self._detectors:
            detector.stop()
