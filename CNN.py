import numpy as np
import torch
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime

startTime = datetime.datetime.now()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Hyper parameters
num_epochs = 200
num_classes = 39
batch_size = 10
learning_rate = 0.000001

transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("Dataset", transform=transform)

indices = list(range(len(dataset)))
split = int(np.floor(0.85 * len(dataset)))  # train_size
validation = int(np.floor(0.70 * split))  # validation
np.random.shuffle(indices)

train_indices, validation_indices, test_indices = (
    indices[:validation],
    indices[validation:split],
    indices[split:],
)

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)

validation_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=validation_sampler
)

test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)


class ConvNet(nn.Module):
    def __init__(self, num_classes=39):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),  # every batch norm adds some learnable parameters 112 in total
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.fc1 = nn.Linear(8 * 8 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(-1, 8 * 8 * 32)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


model = ConvNet(num_classes).to(device)

total_params = sum(param.numel() for param in model.parameters())
print(f'Total Network Parameters: {total_params}')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x_epoch = []
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    training_epoch_loss = 0
    validation_epoch_loss = 0
    correct_v = 0
    total_v = 0
    for i, (train_images, train_labels) in enumerate(train_loader):
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)

        # Forward pass
        train_outputs = model(train_images)
        train_loss = criterion(train_outputs, train_labels)

        training_epoch_loss += train_loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        _, train_predicted = torch.max(train_outputs.data, 1)
        total_v += train_labels.size(0)
        correct_v += (train_predicted == train_labels).sum().item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss:        {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, train_loss.item()))  # printing info every iteration

    x_epoch.append(epoch + 1)
    training_loss.append(training_epoch_loss / len(train_loader))
    training_accuracy.append(correct_v / total_v)

    # Validate the model
    model.eval()
    with torch.no_grad():
        correct_v = 0
        total_v = 0
        for valid_images, valid_labels in validation_loader:
            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)
            valid_outputs = model(valid_images)
            valid_loss = criterion(valid_outputs, valid_labels)

            validation_epoch_loss += valid_loss.item()

            _, valid_predicted = torch.max(valid_outputs.data, 1)
            total_v += valid_labels.size(0)
            correct_v += (valid_predicted == valid_labels).sum().item()

        print('Validation Accuracy of the model in epoch {} : {} %'.format(epoch + 1, 100 * correct_v / total_v))

    validation_loss.append(validation_epoch_loss / len(validation_loader))
    validation_accuracy.append(correct_v / total_v)

# Test the model
model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        outputs = model(test_images)
        _, test_predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (test_predicted == test_labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

endTime = datetime.datetime.now()
print(f'Running time: {endTime - startTime}')

plt.figure()
plt.plot(x_epoch, training_loss, color='b', label='training loss')
plt.plot(x_epoch, validation_loss, color='g', label='validation loss')
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.figure()
plt.plot(x_epoch, training_accuracy, color='b', label='training accuracy')
plt.plot(x_epoch, validation_accuracy, color='g', label='validation accuracy')
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()
