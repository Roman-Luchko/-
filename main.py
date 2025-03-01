#from Code.Data_preparation import data_preparation as dp
import pydicom

# Пример использования
#folder_path = 'HAOS/Test_Sets/CT/7/DICOM_anon'  # Укажи путь к своей папке с DICOM изображениями
#dp.display_dicom_images_from_folder(folder_path)

#from Code.Model_train.utlis import *
from Code.Model_train.attemtion_unet import AttentionUNet
from torchvision.utils import make_grid
from Code.Model_train.train import train_and_test
from Code.Model_train.loss import dice_coeff, FocalLoss
import torch.nn as nn
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
import cv2
import glob


class DicomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dcm')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        dicom_path = self.files[idx]
        dicom_data = pydicom.dcmread(dicom_path)

        image = torch.tensor(dicom_data.pixel_array, dtype=torch.float32)

        # Преобразуем (H, W) -> (1, H, W)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        # Проверяем, есть ли маска (например, если она сохранена в отдельной папке)
        mask_path = dicom_path.replace("images", "masks")  # Убедись, что путь правильный

        if os.path.exists(mask_path):
            mask_data = pydicom.dcmread(mask_path).pixel_array
            mask = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
        else:
            mask = torch.zeros_like(image)  # Если маски нет, создаем пустую

        # Приводим изображение к 3 каналам, если модель требует (3, H, W)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return {"image": image, "mask": mask}


def get_data_loaders(data_dir, batch_size=8, num_workers=0):
    dataset = DicomDataset(data_dir)

    # Разделяем на train (70%), val (15%) и test (15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"training": train_loader, "val": val_loader, "test": test_loader}



data_dir = "HAOS/test"
batch_size = 4
epochs = 100
dataloaders = get_data_loaders(data_dir, batch_size=batch_size)

def train():
    model = AttentionUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = FocalLoss()

    trained_model = train_and_test(model, dataloaders, optimizer, criterion, num_epochs=1)

    return trained_model


def plot_prediction(model, dataloaders):
    if 'val' not in dataloaders or len(dataloaders['val'].dataset) == 0:
        print("Ошибка: валидационный датасет пустой!")
        return

    dataiter = iter(dataloaders['val'])
    try:
        batch = next(dataiter)
    except StopIteration:
        print("Ошибка: нет данных в `val` даталоадере!")
        return

    f = plt.figure(figsize=(20, 20))
    grid_img = make_grid(batch['mask'])
    grid_img = grid_img.permute(1, 2, 0)
    plt.imshow(grid_img)
    plt.title('Ground truth')
    plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = batch['image'].to(device)
    prediction = model(inputs).detach().cpu()

    f = plt.figure(figsize=(20, 20))
    grid_img = make_grid(prediction)
    grid_img = grid_img.permute(1, 2, 0)
    plt.imshow(grid_img)
    plt.title('Prediction')
    plt.show()


trained_model = train()
plot_prediction(trained_model, dataloaders)

#plot_batch_from_dataloader(dataloaders, 4)

'''image = cv2.imread('data/training/images/21_training.tif')
image = cv2.copyMakeBorder(image, top=4, bottom=4, left=6, right=5,
                           borderType=cv2.BORDER_CONSTANT)

img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
clahe = cv2.createCLAHE(clipLimit=2.0)
img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
plt.imshow(img_output)
plt.show()'''