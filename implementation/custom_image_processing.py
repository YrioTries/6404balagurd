"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки NumPy.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Собеля)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import interfaces

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class CustomImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ и углов.

    Методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Sobel).
        corner_detection(image): Обнаруживает углы (Harris).
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.


        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        kernel_height, kernel_width = kernel.shape
        pad_height, pad_width = kernel_height // 2, kernel_width // 2

        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode="constant")

        window_view = sliding_window_view(padded_image, (kernel_height, kernel_width))

        output = np.sum(window_view * kernel, axis=(-2, -1))

        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует функцию cv2.cvtColor для преобразования цветного изображения
        в чёрно-белое.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        red_ch, green_ch, blue_ch = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray_img = red_ch * 0.299 + green_ch * 0.587 + blue_ch * 0.114
        gray_img = gray_img.astype(np.uint8)

        return gray_img

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        inv_gamma = 1.0 / gamma
        image = image.astype(np.float32)

        if image.ndim == 3:
            _, _, channels = image.shape
            for i in range(channels):
                print(i)
                image[:, :, i] = np.power(image[:, :, i] / 255, inv_gamma)

        if image.ndim == 2:
            image = np.power(image / 255, inv_gamma)

        return np.clip(image * 255, 0, 255).astype(np.uint8)

    def _smooth(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Применяет сглаживание к изображению.

        Каждый пиксель изображения становится равен среднему соседних пикселей.
        Args:
            image: (np.ndarray): Входное изображение.
            kernel_size: (int): Размер фильтра в пикселях.

        Returns:
            np.ndarray: Изображение после сглаживания.
        """
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= kernel.size
        return self._convolution(image, kernel)

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Собеля для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        gray = self._rgb_to_grayscale(image).astype(np.float32)
        gray = self._gamma_correction(gray, 2.2)
        edges_x = self._convolution(gray, sobel_x)
        edges_y = self._convolution(gray, sobel_y)

        edges = np.sqrt(edges_x ** 2 + edges_y ** 2)

        edges = self._smooth(edges, 3)

        threshold = np.percentile(edges, 96)

        result = np.zeros_like(edges)
        result[edges > threshold] = 255
        return result.astype(np.uint8)

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.
8
        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """

        k = 0.04  # параметр Харриса

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        gray = self._rgb_to_grayscale(image).astype(np.float32)

        result = image.copy()

        Ix = self._convolution(gray, sobel_x)
        Iy = self._convolution(gray, sobel_y)

        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy

        Ix2 = self._smooth(Ix2, 5)
        Iy2 = self._smooth(Iy2, 5)
        Ixy = self._smooth(Ixy, 5)

        det_M = Ix2 * Iy2 - Ixy ** 2
        trace_M = Ix2 + Iy2
        R = det_M - k * (trace_M ** 2)

        result[R > 0.01 * R.max()] = [255, 0, 0]
        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")