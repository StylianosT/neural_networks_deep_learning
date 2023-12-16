import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("Dataset", transform=transform)

indices = list(range(len(dataset)))
split = int(np.floor(0.85 * len(dataset)))
np.random.shuffle(indices)
train_indices, test_indices = (
    indices[:split],
    indices[split:],
)

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

batch_size = 1
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)

x_train = []
y_train = []
x_test = []
y_test = []

print()
print(f'Train set: {len(train_loader)}')
print(f'Test set:  {len(test_loader)}')
print()

for image, label in train_loader:
    im = image.squeeze().flatten().numpy()
    lb = label.squeeze().flatten().numpy()
    x_train.append(im)
    y_train.append(lb[0])

for image, label in test_loader:
    im = image.squeeze().flatten().numpy()
    lb = label.squeeze().flatten().numpy()
    x_test.append(im)
    y_test.append(lb[0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(x_train, y_train)
score = knn1.score(x_test, y_test)
print(f'1-NN accuracy: {score}')

nc = NearestCentroid()
nc.fit(x_train, y_train)
score2 = nc.score(x_test, y_test)
print(f'NCC accuracy: {score2}')

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train, y_train)
score3 = knn3.score(x_test, y_test)
print(f'3-NN accuracy: {score3}')
print()

if score > score2:
    print("1-NN is more accurate than NCC")
elif score == score2:
    print("1-NN accuracy is equal to NCC")
else:
    print("NCC is more accurate than 1-NN")

if score3 > score2:
    print("3-NN is more accurate than NCC")
elif score3 == score2:
    print("3-NN accuracy is equal to NCC")
else:
    print("NCC is more accurate than 3-NN")

if score3 > score:
    print("3-NN is top")
elif score3 == score:
    print("3-NN and 1-NN are top")
else:
    print("1-NN is top")

print()
print("Train set samples:")
counter = [0] * 39
for i in range(len(y_train)):
    el = y_train[i]
    counter[el] += 1
print(counter)

print()
print("Test set samples:")
counter2 = [0] * 39
for i in range(len(y_test)):
    el = y_test[i]
    counter2[el] += 1
print(counter2)
