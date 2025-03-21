import torch
import pydicom
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from PIL import Image
from Code.Model_train.attemtion_unet import AttentionUNet
from Code.Model_train.train import train  # Заменили train_and_test на вашу train
from Code.Model_train.loss import DiceLoss  # Заменили FocalLoss на ваш BCEWithLogitsLossWrapper

import matplotlib
matplotlib.use('TkAgg')

class DicomDataset(Dataset):
    def __init__(self, data_dir, mask_dir):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.dicom_files = [f for f in os.listdir(data_dir) if f.endswith('.dcm')]
        self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

        if len(self.dicom_files) != len(self.mask_files):
            raise ValueError(
                f"Количество изображений ({len(self.dicom_files)}) и масок ({len(self.mask_files)}) не совпадает!")

        for dicom_file, mask_file in zip(self.dicom_files, self.mask_files):
            dicom_path = os.path.join(data_dir, dicom_file)
            mask_path = os.path.join(mask_dir, mask_file)
            print(f"Изображение: {dicom_path}, Маска: {mask_path}")

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_path = os.path.join(self.data_dir, self.dicom_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        mask = Image.open(mask_path)
        mask = mask.convert("L")  # Преобразуем в черно-белое изображение

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Добавляем канал (C, H, W)
        mask = torch.tensor(np.array(mask) / 255.0, dtype=torch.float32).unsqueeze(0)  # Тоже добавляем канал для маски

        # Генерируем имя для предсказания, например, используя имя изображения с добавлением префикса или суффикса
        predicted_filename = self.dicom_files[idx]# Это пример, замените на нужный формат

        # Возвращаем словарь с именами для изображения, маски и предсказания
        return {
            'image': image,
            'mask': mask,
            'filename': self.dicom_files[idx],  # Имя изображения
            'mask_filename': self.mask_files[idx],  # Имя маски
            'predicted_filename': predicted_filename  # Имя для предсказания
        }


def get_data_loaders(data_dir, mask_dir, batch_size=8, num_workers=0):
    dataset = DicomDataset(data_dir, mask_dir)

    # Разделяем на train (70%), val (15%) и test (15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"training": train_loader, "val": val_loader, "test": test_loader}


data_dir = "HAOS/test/dataset"  # путь к датасету
mask_dir = "HAOS/test/masks"  # путь к маскам
batch_size = 4  # интервал, один же батч
epochs = 100  # количество проходов через датасет
dataloaders = get_data_loaders(data_dir, mask_dir, batch_size=batch_size)  # список загрузчиков
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

#########


# Вызов , передавая даталоадеры


def train_model():
    model_path = "attention_unet.pkl"

    # Создаем модель
    model = AttentionUNet(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DiceLoss()

    # Запуск тренировки
    trained_model= train(
        model, dataloaders, optimizer, criterion, num_epochs=epochs
    )

    # Проверка на NaN/Inf в параметрах модели перед сохранением
    for name, param in trained_model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"⚠️ Параметр {name} содержит NaN/Inf! Остановка сохранения модели.")
            return None, None, None

    # Сохраняем модель
    with open(model_path, "wb") as f:
        pickle.dump(trained_model, f)
    print("✅ Модель сохранена.")

    return trained_model


trained_model = train_model()


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageMaskNavigator:
    def __init__(self, dataloaders, model):
        self.dataloaders = dataloaders
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.index = 0  # Индекс текущей пары изображения и маски
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))  # Три изображения (изображение, маска, предсказание)

        # Подключаем обработчик событий для перелистывания
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.display_current_image_mask_prediction()
        plt.show()

    def on_key(self, event):
        """Обработчик нажатий клавиш для перелистывания изображений"""
        if event.key == 'right':  # Перелистать изображение вправо (следующее)
            self.index = (self.index + 1) % self.get_total_images()  # Цикличность
            self.display_current_image_mask_prediction()
        elif event.key == 'left':  # Перелистать изображение влево (предыдущее)
            self.index = (self.index - 1) % self.get_total_images()  # Цикличность
            self.display_current_image_mask_prediction()

    def get_total_images(self):
        """Получить общее количество изображений в даталоадерах"""
        total_images = 0
        for dataloader in self.dataloaders.values():
            for batch in dataloader:
                total_images += len(batch['image'])
        return total_images

    def display_current_image_mask_prediction(self):
        """Отображает текущую пару изображения, маски и предсказания"""
        image_counter = 0
        for dataset_name, dataloader in self.dataloaders.items():
            for batch in dataloader:
                for i, (img_name, mask_name) in enumerate(zip(batch['filename'], batch['mask_filename'])):
                    if image_counter == self.index:
                        # Извлекаем текущее изображение и маску
                        current_image = batch['image'][i, 0].numpy()
                        current_mask = batch['mask'][i, 0].numpy()
                        image_title = img_name
                        mask_title = mask_name

                        # Для предсказания используем модель
                        image_tensor = batch['image'][i, 0].unsqueeze(0).unsqueeze(0).to(
                            self.device)  # Добавляем размерность batch и channels

                        # Получаем предсказание
                        with torch.no_grad():  # Выключаем вычисление градиентов
                            prediction = self.model(
                                image_tensor).detach().cpu().squeeze().numpy()  # Получаем предсказание

                        prediction_title = img_name  # Имя предсказания соответствует имени изображения
                        break
                    image_counter += 1
                if 'current_image' in locals():
                    break
            if 'current_image' in locals():
                break

        # Отображаем изображение, маску и предсказание
        self.axes[0].imshow(current_image, cmap="gray")
        self.axes[0].set_title(f"Image: {image_title}", fontsize=10)
        self.axes[0].axis('off')  # Скрыть оси

        self.axes[1].imshow(current_mask, cmap="gray")
        self.axes[1].set_title(f"Mask: {mask_title}", fontsize=10)
        self.axes[1].axis('off')  # Скрыть оси

        self.axes[2].imshow(prediction, cmap="gray")
        self.axes[2].set_title(f"Prediction: {prediction_title}", fontsize=10)
        self.axes[2].axis('off')  # Скрыть оси

        self.fig.canvas.draw()  # Обновляем изображение на экране


# Пример использования:
# dataloaders - ваш dataloader, model - ваша обученная модель, device - устройство (например, 'cuda' или 'cpu')
navigator = ImageMaskNavigator(dataloaders, trained_model)

