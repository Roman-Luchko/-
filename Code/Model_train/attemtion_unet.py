
# Файл для создания самой модели(блоко модели и функции для прямого прохода).

import torch
import torch.nn as nn # готовые слои для нейросетей.


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        # in_channels : входное количество матриц
        # out_channels : выходное количество матриц

        super(ConvBlock, self).__init__()
        #Если ты не вызовешь super().__init__(), PyTorch не сможет отслеживать
        # параметры модели, и она не будет работать.


        # nn.Sequential(...) позволяет объединить слои в один модуль,
        # чтобы потом вызвать их одним вызовом (без явного указания каждого слоя).
        self.conv = nn.Sequential(

            # 1 Первый слой
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # Conv2d : Сверточный слой
            # in_channels : вход
            # out_channels : выход
            # kernel_size : равномерность nxn фильтра
            # stride : шаг фильтра
            # padding : добавляет по краям 1 слой пикселей, для избежани ошибки с
            # с фильтром
            # bias=True : добавляет обучаемый параметр смещения

            # Batch Normalization - метод нормализации
            # Нормализация служит для ускорения обучения

            nn.BatchNorm2d(out_channels),
            # BatchNorm2d нормализует каждый канал отдельно,
            # поэтому он должен знать, сколько их будет

            # функция актвации ReLu
            nn.ReLU(inplace=True),
            # inplace=False (по умолчанию) — создаёт новый тензор с
            # результатами преобразования.
            # inplace=True — изменяет входной тензор на месте, без создания копии.

            # Когда использовать inplace=True?
            # - Экономит память (нет лишнего выделения).
            # - Иногда ускоряет вычисления.
            # - Нельзя использовать, если слой стоит до слоёв, которым нужен оригинальный тензор
            #   (например, в некоторых сложных графах вычислений).

            # 2 Слой, не обьясняю, выше в коде похожий словй
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # прямое распространение (forward pass)(или максимально по простому
    # слева->направо по схеме нейронки
    def forward(self, x):
        x = self.conv(x)
        # conv : есть уже в коде, при вызове conv все что вложено
        # с помощью nn.Sequential, поочередно используеться
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.Upsample(scale_factor=2) – увеличивает размер входного тензора
            # в 2 раза.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# по сути, просто слой для обработки "skip connection"
# с целью убрать ненужные признаки
class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, n_coefficients):
        # F_g : тоже же самое,что и in_channels
        # F_l : тоже же самое,что и out_channels
        # n_coefficients : сколько каналов будет
        # в скрытом слое на выходе, зачем это я не знаю
        super(AttentionBlock, self).__init__()

        #В общем,есть два блока(W_g,W_X), которые потом обьединяют свой результат
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):

        # skip_connection (X) → содержит старые признаки,
        # которые были извлечены на пути вниз (в энкодере),
        # до того как данные стали меньше через max pooling.
        # Эти признаки несут мелкие детали изображения.

        # gate (G) → содержит новые признаки,
        # которые были обработаны на пути вверх (в декодере).
        # Они несут высокоуровневую информацию после нескольких свёрточных слоёв.

        # Задача Attention — выбрать наиболее важные признаки
        # из skip_connection на основе gate.
        # Это помогает убрать лишний шум и оставить только полезную информацию.

        # Нахождение W_g, W_x
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        #по схеме дальше идут 2 слоя
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        #для уселение важных признаков,все по схеме
        out = skip_connection * psi

        return out


# Сделать уже полноценную модель из готовых блоков в коде выше
# ну или есть и блоки с библиотеки
class AttentionUNet(nn.Module):

    def __init__(self, img_ch=1, output_ch=1):
        # img_ch : то же самое,что и in_channels
        # output_ch : то же самое,что и out_channels
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # nn.MaxPool2d — это макспулинг (max pooling), который уменьшает размер изображения, выделяя наиболее значимые признаки.
        # kernel_size=2 — окно 2×2, берёт максимум из каждой такой области.
        # stride=2 — сдвигается на 2 пикселя, уменьшая размер карты признаков в 2 раза.
        # Он содаеться всего раз, дальше его просто применяют

        # Создание блоков, все как по схеме(Тут Энкодеры)
        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # Создание блоков, всё как по схеме (тут декодер).
        # В декодере в U-Net, кроме "обработки-изменения размеров", есть
        # и skip connections (просто передача информации). В Attention U-Net
        # эту информацию дополнительно обрабатывают с помощью
        # ещё одного слоя для улучшения работы модели.


        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):


        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)


        # gate (G) — это сигнал от более глубокого слоя (из декодера).
        # Этот сигнал несёт основную информацию о текущем состоянии обработки данных.
        # skip_connection (X) — это данные из энкодера (из соответствующего уровня).
        # Эти данные были сохранены перед пулингом и помогают восстановить детали изображения.


        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        # torch.cat() — объединяет тензоры вдоль указанного измерения (dim).
        # s4 : это обработанный skip_connection после Attention блока.
        # d5 : это выход из предыдущего апскейлинга (UpConv).
        # dim=1 : dim=1 означает, что мы объединяем тензоры по
        # канальному измерению, то есть добавляем новые фильтры (признаки).
        # не понял,что делает dim, кроме обьединения

        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out