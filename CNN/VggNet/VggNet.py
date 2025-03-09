import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchsummary
import matplotlib.pyplot as plt
import json
from contextlib import redirect_stdout

net_config = json.load(open('Config.json', 'r'))

pre_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # flip image randomly
    transforms.RandomHorizontalFlip(),
    # crop image randomly
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
])

train_data = torchvision.datasets.CIFAR100(root='../data', train=True, transform=pre_transform,
                                           download=True)
test_data = torchvision.datasets.CIFAR100(root='../data', train=False, transform=pre_transform,
                                          download=True)

train_iter = torch.utils.data.DataLoader(
    train_data, batch_size=net_config["batch_size"], shuffle=True, num_workers=net_config["workers"])
test_iter = torch.utils.data.DataLoader(
    test_data, batch_size=net_config["batch_size"], shuffle=False, num_workers=net_config["workers"])


VggConfigs = {
    "A": [[64, 1], [128, 1], [256, 2], [512, 2], [512, 2]]
}


class VggNet(nn.Module):

    def __init__(self, config_name):
        super(VggNet, self).__init__()
        self.net = nn.Sequential()
        self.build_net_from_config(VggConfigs[config_name])

    def build_net_from_config(self, config):
        in_channels = 3
        for block_cfg in config:
            out_channels, num_convs = block_cfg[0], block_cfg[1]
            self.net.append(
                self.create_vgg_block(in_channels, out_channels, num_convs))
            in_channels = out_channels

        self.net.append(nn.Flatten())

        self.net.append(nn.Dropout(net_config["dropout"]))
        self.net.append(nn.Linear(512 * 7 * 7, 4096))
        self.net.append(nn.ReLU())

        self.net.append(nn.Dropout(net_config["dropout"]))
        self.net.append(nn.Linear(4096, 4096))
        self.net.append(nn.ReLU())

        self.net.append(nn.Linear(4096, 100))

    def create_vgg_block(self, in_channels, out_channels, num_convs, has_bn=False):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


def init_weight(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=0.01)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Use device: {device}')

net = VggNet("A").to(device)
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

        if i % 100 == 0:
            print(f'\t{i}/{len(train_iter)}')

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

    # update learning rate
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
