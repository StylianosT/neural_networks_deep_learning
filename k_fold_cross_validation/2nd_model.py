# convolutional and fully connected with grayscale and one hidden layer
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import KFold


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=39):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, 39)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(-1, 7 * 7 * 32)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Hyper parameters
num_epochs = 5
num_classes = 39
batch_size = 100
learning_rate = 0.001
k_folds = 5
criterion = nn.CrossEntropyLoss()

# For fold results
results = {}

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("Dataset", transform=transform)

indices = list(range(len(dataset)))
split = int(np.floor(0.85 * len(dataset)))  # train_size

train_indices, test_indices = (
    indices[:split],
    indices[split:],
)

k_fold = KFold(n_splits=5, shuffle=True, random_state=1)

for fold, (train_indices, test_indices) in enumerate(k_fold.split(dataset)):
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler
    )

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    print('Training process has finished.')

    # Test the model
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
        results[fold] = 100.0 * (correct / total)

print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum / len(results.items())} %')