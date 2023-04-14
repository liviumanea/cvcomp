import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from cvcomp.models.base import CVModel

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d4/1")


class EffdetB4(CVModel):

    def __init__(self, confidence_threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._confidence_threshold = confidence_threshold
        self._detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d4/1")

    def get_name(self) -> str:
        return "EffdetB4"

    def detect(self, frame: np.ndarray) -> np.ndarray:
        boxes = self._get_bounding_boxes(frame)
        return self._draw_bounding_boxes(frame, boxes)

    def _get_bounding_boxes(self, frame: np.ndarray):
        shape = frame.shape
        image_tensor = tf.convert_to_tensor(frame)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        detections = self._detector(image_tensor)
        boxes = (
                detections["detection_boxes"].numpy()[0] * np.array([shape[0], shape[1], shape[0], shape[1]])
        ).astype(int)
        scores = detections["detection_scores"].numpy()[0]
        labels = detections["detection_classes"].numpy()[0].astype(int)
        return zip(boxes, scores, labels)

    def _draw_bounding_boxes(self, frame: np.ndarray, boxes):
        for box, score, label in boxes:
            if score > self._confidence_threshold:
                y1, x1, y2, x2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Label: {label}, Score: {score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
