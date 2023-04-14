import cv2
import numpy as np

from cvcomp.models.helpers import get_detection_classes
from cvcomp.models.base import CVModel


class YoloV4(CVModel):

    def __init__(
            self,
            config_path: str,
            weights_path: str,
            classes: str,
            conf_threshold=0.6,
            nms_threshold=0.4,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self._classes = get_detection_classes(classes)
        self._layer_names = self._net.getLayerNames()
        self._output_layers = [self._layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold

    def get_name(self) -> str:
        return "YoloV4"

    def detect(self, frame: np.ndarray) -> np.ndarray:
        height, width, channels = frame.shape

        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Set the blob as input
        self._net.setInput(blob)

        # Run inference through the network
        outs = self._net.forward(self._output_layers)

        # Get the bounding boxes
        class_ids, confidences, boxes = self._get_bounding_boxes(outs, height, width)

        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self._conf_threshold, self._nms_threshold)

        # Draw the bounding boxes
        self._draw_bounding_boxes(frame, indices, boxes, confidences, class_ids)

        return frame

    def _get_bounding_boxes(self, outs, height, width):
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self._conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return class_ids, confidences, boxes

    def _draw_bounding_boxes(self, frame, indices, boxes, confidences, class_ids):

        for i in indices:
            x, y, box_width, box_height = boxes[i]
            label = f"{self._classes[class_ids[i]]} {confidences[i]:.2f}"

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)

            # Add the label to the box
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
