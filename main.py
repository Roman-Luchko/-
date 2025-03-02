#DICOM — это стандарт, используемый в медицине для хранения,
#              передачи и обработки медицинских изображений,
#              таких как рентгеновские снимки, МРТ, КТ и УЗИ.

#общедоступные библиотеки

import pydicom # для работы с медицинскими изображениями в формате dicom
import os # для работы с операционной системой
import torch # для машинного обучения и глубокого обучения
import matplotlib.pyplot as plt # для визуализации данных
from torch.utils.data import Dataset, DataLoader, random_split  # для работы с данными при обучении нейросетей.
from torchvision.utils import make_grid # для удобной визуализации нескольких изображений в одном холсте

# пользовательские файлы
from Code.Model_train.attemtion_unet import AttentionUNet # сама модель
from Code.Model_train.train import train_and_test # ее обучение
from Code.Model_train.loss import FocalLoss # не смотрел, но loss єто ошибка

# Пример использования
#folder_path = 'HAOS/Test_Sets/CT/7/DICOM_anon'  # Укажи путь к своей папке с DICOM изображениями
#dp.display_dicom_images_from_folder(folder_path)


class DicomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir # путь к датасету
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dcm')]
        # self.files : это список путей ко всем DICOM-файлам
        # os.listdir(data_dir) : Получает список всех файлов и папок в data_dir.
        # for f in os.listdir(data_dir) : Проходит по каждому файлу f в этой папке.
        # if f.endswith('.dcm') : Оставляет только файлы, которые заканчиваются на .dcm.
        # os.path.join(data_dir, f) : Объединяет путь data_dir и имя файла f, создавая полный путь.
        # ["file1.dcm", "file2.dcm", "file3.jpg", "file4.dcm"] - пример выхода
    def __len__(self): # количества обьектов в датасете
        return len(self.files)

    def __getitem__(self, idx): # вернуть один елемент из датасета по индеску
        dicom_path = self.files[idx] # получить путь обьекта по индекску
        dicom_data = pydicom.dcmread(dicom_path) # загрузка dcim файла

        # Тензор — это обобщение понятий скаляра, вектора и матрицы.
        # В машинном обучении и PyTorch тензоры — это основная структура данных,
        # похожая на NumPy-массив, но с поддержкой работы на GPU.
        image = torch.tensor(dicom_data.pixel_array, dtype=torch.float32)
        #torch.tensor : получить изображение из dcim в тензор
        # pixel_array : получить изображение dicom_data в формате Numpy
        # dtype=torch.float32 : приводит значения к float32, для нормальной работы
        # нейронки

        # Преобразуем (H, W) -> (1, H, W)
        # Этот код изменяет форму тензора, добавляя новый измерение канала (C)
        # перед высотой (H) и шириной (W).
        # Канал — это отдельный слой данных в изображении,
        # содержащий информацию о разных аспектах пикселей, таких как цвет,
        # глубина или другие признаки.
        if len(image.shape) == 2:      # .shape — это атрибут тензора, который показывает его количество осей
            image = image.unsqueeze(0) # unsqueeze(dim) добавляет новую ось (измерение)
                                       # в тензоре по указанному индексу dim

        # Проверяем, есть ли маска (например, если она сохранена в отдельной папке)
        mask_path = dicom_path.replace("images", "masks")
        # Этот код создаёт путь к файлу с маской,
        # если он хранится в папке "masks/", вместо "images/"

        if os.path.exists(mask_path): # проверяет, существует ли файл по указанному пути.
            mask_data = pydicom.dcmread(mask_path).pixel_array
            # pydicom.dcmread(mask_path)
            mask = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
            # .pixel_array получает изображение маски в виде массива пикселей (обычно NumPy).
            # .unsqueeze(0) добавляет новое измерение (канал C) в начало тензора.
            # Это нужно, чтобы привести маску к формату (C, H, W),
            # который ожидают нейросети.(Я не знаю , зачем этот кусок кода здесь).
        else:
            mask = torch.zeros_like(image)

        # Приводим изображение к 3 каналам, если модель требует (3, H, W)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
            #torch.zeros_like(image) создаёт пустую маску
            # такого же размера, как image,
            # но заполненную нулями.

        return {"image": image, "mask": mask}
        #Возразает фото и маску для него


# Эта функция создаёт загрузчики данных (DataLoaders) для обучения нейросети.
def get_data_loaders(data_dir, batch_size=8, num_workers=0):
    dataset = DicomDataset(data_dir)
    # создаёт объект класса DicomDataset из указазаного изоражения

    # Разделяем на train (70%), val (15%) и test (15%)
    # определение размеров
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # батчи - єто когда между обновлеяниями весами ставят интервал не целая епоха,
    # а определенный размер
    # DataLoader загружает не все данные сразу, а по мере работы нейросети.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # dataset : Сам датасет
    # batch_size : Сколько изображений загружается за 1 шаг
    # shuffle=True/False : Нужно ли перемешивать данные перед каждой эпохой
    # num_workers : Количество потоков для загрузки (0 = в главном потоке)

    return {"training": train_loader, "val": val_loader, "test": test_loader}



data_dir = "HAOS/test" # путь датасета
batch_size = 4 # интервал,оне же батч
epochs = 100 # количество проходов через датасет
dataloaders = get_data_loaders(data_dir, batch_size=batch_size) # список загрузщиков

def train(): # тренировка самой модели
    model = AttentionUNet() # создание самой модели

    # Оптимизатор нужен, чтобы автоматизировать
    # и ускорить процесс обновления весов, используя градиенты.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # optimizer = torch.optim.<OPTIMIZER>(model.parameters(), lr=0.001, **kwargs)
    # <OPTIMIZER> — название алгоритма оптимизации (SGD, Adam, RMSprop и т. д.).
    # model.parameters() — передаёт веса модели, которые нужно обновлять.
    # lr (learning rate) — скорость обучения ЦЕЛОЙ нейронной сети.
    # **kwargs — дополнительные параметры (например, momentum, weight_decay).
    criterion = FocalLoss()
    # Эта строка создаёт функцию потерь (loss function) — FocalLoss().
    # Она используется для обучения нейросети,
    # чтобы измерять, насколько сильно текущие
    # предсказания отличаются от правильных меток.

    #train_and_test - пользовательская функция для обучения
    trained_model = train_and_test(model, dataloaders, optimizer, criterion, num_epochs=1)
    # model - сама модель
    # dataloaders -  загрузчики
    # optimizer - оптимизатор
    # criterion - функция потерь
    # num_epochs - количество эпох
    return trained_model


def plot_prediction(model, dataloaders):
    # мне тут лень понимать, пока єто не важно
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