import numpy as np

from cvcomp.models.base import CVModel


class Noop(CVModel):

    def get_name(self) -> str:
        return "noop"

    def detect(self, frame: np.ndarray) -> np.ndarray:
        return frame
