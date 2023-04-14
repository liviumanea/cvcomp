import cv2

import config
from cvcomp.io import MultiprocessDetector, ModelMuxer


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


def main():

    mp_yolo = MultiprocessDetector(
        "YoloV4",
        make_yolo,
        capacity=1,
    )

    mp_effdet = MultiprocessDetector(
        "EffdetB4",
        make_effdet,
        capacity=1,
    )

    detector = ModelMuxer(
        [mp_yolo, mp_effdet],
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
