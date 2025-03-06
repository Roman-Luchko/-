import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def train(model, dataloaders, optimizer, criterion, num_epochs=3, show_images=False):
    since = time.time()

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, num_epochs + 1):

        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        batch_train_loss = 0.0

        model.train()  # Set model to training mode

        for sample in iter(dataloaders['training']):

            if show_images:
                grid_img = make_grid(sample['image'])
                grid_img = grid_img.permute(1, 2, 0)
                plt.imshow(grid_img)
                plt.show()

            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                # Проверка значений предсказаний и масок
                print(f"Output Min: {outputs.min().item()}, Max: {outputs.max().item()}")
                print(f"Mask Min: {masks.min().item()}, Max: {masks.max().item()}")

                loss = criterion(outputs, masks)

                # Проверка на NaN или Inf в потере
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Loss is NaN or Inf")

                # Отладка для Dice Loss
                print(f"Loss: {loss.item()}")

                loss.backward()

                # Проверка градиентов
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"⚠️ NaN gradient in {name}")
                        elif (param.grad == 0).all():
                            print(f"⚠️ Zero gradient in {name}")

                optimizer.step()

                batch_train_loss += loss.item() * sample['image'].size(0)

        epoch_train_loss = batch_train_loss / len(dataloaders['training'].dataset)
        print(f'training Loss: {epoch_train_loss:.4f}')

    print(f'Training completed in {time.time() - since:.0f}s')

    return model  # Возвращаем только обученную модель
