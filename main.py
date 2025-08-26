import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
n = 4
image_size = 64  # Размер изображения
learning_rate = 1e-4
epochs = 50
batch_size = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for img_name in os.listdir(data_dir):
            self.image_paths.append(os.path.join(data_dir, img_name))
            l_bracket = self.image_paths[-1].find('[')
            r_bracket = self.image_paths[-1].find(']')
            label_str = self.image_paths[-1][l_bracket + 1: r_bracket]
            label_str = label_str.replace('.', '')
            targets = list(map(int, label_str.split()))
            targets = [x - 1 for x in targets]
            targets = torch.tensor(targets, dtype=torch.int64)
            # targets = F.one_hot(targets, 3)
            targets = targets.float()
            self.labels.append(targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")
        torch.set_printoptions(profile="full")
        if self.transform:
            image = self.transform(image)

        return image, label


class ImageFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, image_size=image_size, n=n):
        self.n = n
        self.image_size = image_size
        super(ImageFeatureExtractor, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64*16*16, 10000)
        self.fc2 = nn.Linear(10000, 800)
        self.fc3 = nn.Linear(800, self.n)
    def softmax_n(self, x, n):
        exp_x = torch.exp(x - torch.max(x))
        softmax = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
        return sum(np.arange(n)) * softmax
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax_n(x, self.n)
        return x


# 6. Создание экземпляра модели
model = ImageFeatureExtractor(input_channels=1, image_size=image_size, n=n)
# model = ImageFeatureExtractor(input_channels=1, image_size=image_size, n=n).to(device)
# 7. Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


dataset = TestDataset(data_dir='D:\\учеба\\maze_problem\\Dataset_n4_l4', transform=transforms.Compose(
    [       transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Инверсия: чёрные линии -> 1.0, белый фон -> 0.0
            transforms.Lambda(lambda x: 1 - x),
            # Бинаризация (если нужно гарантировать 0/1)
            # transforms.Lambda(lambda x: (x > 0.1).float())
     ]
))
train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 10. Обучение модели
train_losses = []  # Для хранения loss на обучающей выборке
test_losses = []   # Для хранения loss на тестовой выборке
train_acc = []
test_acc = []

for epoch in range(epochs):
    model.train()
    cur_train_loss = 0
    cur_test_loss = 0
    acc_train = 0
    acc_test = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        cur_train_loss += loss.item()
        predicted_vectors, pred_ind = torch.sort(outputs, descending=True)
        true_vectors, true_ind = torch.sort(targets, descending=True)
        accuracy = (pred_ind == true_ind).float().mean()
        acc_train += accuracy.item()
    avg_train_loss = cur_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    accuracy_train = acc_train / len(train_loader)
    train_acc.append(accuracy_train)

    # Оценка на тестовой выборке
    running_test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            # x, y = x.to(device), y.to(device)
            test_outputs = model(x) # Вычисляем выход на тестовой выборке
            test_loss = criterion(test_outputs, y)  # Вычисляем loss на тестовой выборке
            cur_test_loss += test_loss.item()
            predicted_vectors, pred_ind = torch.sort(test_outputs, descending=True)
            true_vectors, true_ind = torch.sort(y, descending=True)
            accuracy = (pred_ind == true_ind).float().mean()
            acc_test += accuracy.item()
        avg_test_loss = cur_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy_test = acc_test/len(test_loader)
        test_acc.append(accuracy_test)
        # if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
              f"Train Acc: {accuracy_train:.4f}, Test Acc: {accuracy_test:.4f}")

# 11. Тестирование модели (вывод результатов)
model.eval()

with torch.no_grad():
    for x, y in test_loader:
        # x, y = x.to(device), y.to(device)
        test_output = model(x)
        print("Original:\n", y)
        print("Predicted:\n", test_output)
        print('-----')

# 12. Визуализация графиков
plt.figure(1, figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.grid(True)

plt.figure(2, figsize=(10, 5))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Curve')
plt.legend()
plt.grid(True)
plt.show()