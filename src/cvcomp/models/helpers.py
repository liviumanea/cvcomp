def get_detection_classes(classes_file_path: str) -> list[str]:
    with open(classes_file_path, "r") as f:
        return [line.strip() for line in f.readlines()]
