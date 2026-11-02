import interfaces
import numpy as np
import cv2

class ImageProcessing(interfaces.IImageProcessing):

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, kernel)

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        gray = self._rgb_to_grayscale(image)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        gray = self._rgb_to_grayscale(image)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        result = image.copy()
        result[dst > 0.01 * dst.max()] = [255, 0, 0]
        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")