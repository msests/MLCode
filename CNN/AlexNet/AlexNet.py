import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchsummary
import json


from matplotlib import pyplot as plt
from contextlib import redirect_stdout

import sys
import os
sys.path.append(os.path.abspath('..'))
from MoreDatasets import ImageNet100  # NOQA: E402


net_config = json.load(open('Config.json', 'r'))


pre_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # flip image randomly
    transforms.RandomHorizontalFlip(),
    # crop image randomly
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
])


train_data = ImageNet100(train=True, transform=pre_transform)
test_data = ImageNet100(train=False, transform=pre_transform)

train_iter = torch.utils.data.DataLoader(
    train_data, batch_size=net_config["batch_size"], shuffle=True, num_workers=net_config["workers"])
test_iter = torch.utils.data.DataLoader(
    test_data, batch_size=net_config["batch_size"], shuffle=False, num_workers=net_config["workers"])

print("Dataset loaded.")


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(),

            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, 100),
        )

    def forward(self, x):
        return self.net(x)


def init_weight(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Use device: {device}')

net = AlexNet().to(device)
net.apply(init_weight)

loss = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=net_config["lr"],
    weight_decay=net_config["weight_decay"],
    momentum=net_config["momentum"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, patience=3, factor=0.1, mode="min")

with open('NetSummary.txt', 'w') as f:
    with redirect_stdout(f):
        torchsummary.summary(net, (3, 224, 224))

train_losses = []
test_losses = []
accuriacies = []

epochs = net_config['epochs']
print(f'Begin training..., total epochs: {epochs}')

need_reduce_lr = False

for epoch in range(epochs):
    print(f'epoch: {epoch}')

    net.train()

    total_train_loss = 0
    total_train_iter = 0
    for i, (x, y) in enumerate(train_iter):
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        loss_ = loss(y_hat, y)
        total_train_loss += loss_.item()
        total_train_iter += 1

        # backpropagation and update weights
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

    # calculate mean train loss over all iterations.
    train_loss = total_train_loss / total_train_iter
    train_losses.append(train_loss)
    print(f'\ttrain loss: {train_loss}')

    # evaluate model using current epoch's weights
    net.eval()

    total_test_loss = 0
    total_test_iter = 0
    acc_count = 0
    for i, (x, y) in enumerate(test_iter):
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        loss_ = loss(y_hat, y)
        total_test_loss += loss_.item()
        total_test_iter += 1
        acc_count += (y_hat.argmax(1) == y).sum().item()

    # calculate mean test loss over all iterations.
    test_loss = total_test_loss / total_test_iter
    test_losses.append(test_loss)

    # calculate accuracy on test set
    accuracy = acc_count / len(test_data)
    accuriacies.append(accuracy)

    print(f'\ttest loss: {test_loss}, accuracy: {accuracy}')

    scheduler.step(test_loss)

# output epoch,train_losses,test_losses and accuriacies to "TrainResult.csv"
with open('TrainResult.csv', 'w') as f:
    f.write('epoch,train_loss,test_loss,accuracy\n')
    for i in range(len(train_losses)):
        f.write(str(i) + ',' +
                str(train_losses[i]) + ',' + str(test_losses[i]) + ',' + str(accuriacies[i]) + '\n')


# draw loss chart
plt.plot(train_losses)
plt.plot(test_losses)
plt.scatter(range(len(train_losses)), train_losses, s=30, marker='o')
plt.scatter(range(len(test_losses)), test_losses, s=30, marker='s')
plt.legend(['train loss', 'test loss'])
plt.savefig('LossChart.png')
plt.close()

# draw accuracy chart
plt.plot(accuriacies)
plt.scatter(range(len(accuriacies)), accuriacies, s=30, marker='o')
plt.legend(['accuracy'])
plt.savefig('AccuracyChart.png')
plt.close()
