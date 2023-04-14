import cv2

import config
from cvcomp.io import MultiprocessDetector, ModelMuxer
from cvcomp.io import RedisFrameDispatcher
import multiprocessing


def make_yolo():
    from cvcomp.models.yolov4 import YoloV4
    return YoloV4(
        './data/cfg/yolov4.cfg',
        './data/weights/yolov4.weights',
        './data/coco.names',
    )


def make_effdet():
    from cvcomp.models.effdet import EffdetB4
    return EffdetB4(
        confidence_threshold=0.5,
    )


def run_consumer():
    from cvcomp.io import RedisFrameConsumer
    import time

    consumer = RedisFrameConsumer(
        model_name="YoloV4",
        model_factory=make_yolo,
        redis_url="redis://localhost:6379/0",
    )

    print("started the YoloV4 consumer")
    # TODO: make it a blocking call and find a way to stop it
    time.sleep(600)


def main():

    consumer = multiprocessing.Process(target=run_consumer)
    consumer.start()

    yolo = RedisFrameDispatcher(
        model_name="YoloV4",
        redis_url="redis://localhost:6379/0",
    )

    effdet = MultiprocessDetector(
        "EffdetB4",
        make_effdet,
        capacity=1,
    )

    detector = ModelMuxer(
        [yolo, effdet],
        "Muxer"
    )

    cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prev_frame = detector.get_result(block=False)
        if prev_frame is not None:
            cv2.imshow('frame', prev_frame)

        if detector.is_busy():
            continue

        detector.ingest(frame)

        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.stop()


if __name__ == '__main__':
    main()
