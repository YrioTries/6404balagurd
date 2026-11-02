from abc import ABC, abstractmethod

import numpy as np

class IImageProcessing(ABC):
    @abstractmethod
    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        pass

    @abstractmethod
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        pass