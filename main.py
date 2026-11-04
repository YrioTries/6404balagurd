"""
main.py

Лабораторная работа №1 по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки NumPy.

Автор: Балагур Дмитрий
"""

import argparse
import os

import imageio.v3 as iio

from implementation import ImageProcessing
from implementation import CustomImageProcessing


def main() -> None:
    """
    Консольный интерфейс для применения различных методов обработки к изображению:
    - обнаружение границ (edges)
    - обнаружение углов (corners)

    Запуск:
        python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]

    Аргументы:
        метод: edges | corners | circles
        путь_к_изображению: путь к входному изображению
        -o, --output: путь для сохранения результата (по умолчанию: <имя_входного_файла>_result.png)

    Пример:
        python main.py edges input.jpg
        python main.py corners input.jpg -o corners_result.png
    """
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing")

    parser.add_argument("method", choices=["edges", "corners", "circles"],
                        help="Метод обработки: edges, corners, circles")
    parser.add_argument("input",
                        help="Путь к входному изображению")
    parser.add_argument("-o", "--output",
                        help="Путь для сохранения результата (по умолчанию: <input>_result.png)")
    parser.add_argument("-i", "--impl",
                        choices=["custom", "lib"], 
                        default="lib",
                        help="Путь для сохранения результата (по умолчанию: <input>_result.png)")

    args = parser.parse_args()

    img = iio.imread(args.input)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    if (args.impl == "lib"):
        processor = ImageProcessing()
    else:
        processor = CustomImageProcessing()

    if args.method == "edges":
        result = processor.edge_detection(img)
    elif args.method == "corners":
        result = processor.corner_detection(img)
    elif args.method == "circles":
        result = processor.circle_detection(img)
    else:
        print("Ошибка: неизвестный метод")
        return

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_result.png"

    iio.imwrite(uri=output_path, image=result)
    print(f"Результат сохранен в {output_path}")


if __name__ == "__main__":
    main()