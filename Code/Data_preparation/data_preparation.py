import pydicom
from PIL import Image
import numpy as np
import os


def display_dicom_images_from_folder(folder_path, indices=[0, 1, 2]):
    """
    Отображает три изображения из DICOM файлов в указанной папке.

    :param folder_path: Путь к папке с DICOM файлами.
    :param indices: Список индексов изображений для отображения.
    """
    # Получаем список всех DICOM файлов в папке
    dicom_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')]

    # Проверяем, что в папке достаточно файлов
    if len(dicom_files) < max(indices) + 1:
        print(f"В папке меньше {max(indices) + 1} DICOM файлов.")
        return

    # Отображаем указанные DICOM изображения
    for idx in indices:
        dicom_path = os.path.join(folder_path, dicom_files[idx])

        # Загружаем DICOM файл
        dicom_data = pydicom.dcmread(dicom_path)

        # Извлекаем изображение из DICOM
        image_array = dicom_data.pixel_array

        # Нормализуем изображение (иногда DICOM может содержать значения вне стандартного диапазона 0-255)
        image_array = image_array - np.min(image_array)
        image_array = image_array / np.max(image_array) * 255
        image_array = image_array.astype(np.uint8)  # Преобразуем в целочисленный тип для отображения

        # Преобразуем в изображение с использованием Pillow
        img = Image.fromarray(image_array)

        # Показываем изображение
        img.show(title=f"Image {dicom_files[idx]}")