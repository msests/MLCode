import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchsummary
from matplotlib import pyplot as plt
from contextlib import redirect_stdout

pre_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=pre_transform,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=pre_transform,
                                         download=True)

train_iter = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True)
test_iter = torch.utils.data.DataLoader(
    test_data, batch_size=64, shuffle=False)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=5, stride=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),

            nn.Linear(120, 84),
            nn.Sigmoid(),

            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


def init_weight(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = LeNet().to(device)
net.apply(init_weight)

loss = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

with open('NetSummary.txt', 'w') as f:
    with redirect_stdout(f):
        torchsummary.summary(net, (3, 28, 28))

train_losses = []
test_losses = []
accuriacies = []

epochs = 20
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

        # 反向传播和权重更新
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
    # 计算这一轮的平均训练损失
    train_loss = total_train_loss / total_train_iter
    train_losses.append(train_loss)
    print(f'\ttrain loss: {train_loss}')

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
    # 计算这一轮的平均测试损失
    test_loss = total_test_loss / total_test_iter
    test_losses.append(test_loss)
    # 计算在测试集上的准确度
    accuracy = acc_count / len(test_data)
    accuriacies.append(accuracy)
    print(f'\ttest loss: {test_loss}, accuracy: {accuracy}')

# 把epoch,train_losses，test_losses和accuriacies输出到"TrainResult.csv"
with open('TrainResult.csv', 'w') as f:
    f.write('epoch,train_loss,test_loss,accuracy\n')
    for i in range(len(train_losses)):
        f.write(str(i) + ',' +
                str(train_losses[i]) + ',' + str(test_losses[i]) + ',' + str(accuriacies[i]) + '\n')


# 绘制损失折线图
plt.plot(train_losses)
plt.plot(test_losses)
plt.scatter(range(len(train_losses)), train_losses, s=30, marker='o')
plt.scatter(range(len(test_losses)), test_losses, s=30, marker='s')
plt.legend(['train loss', 'test loss'])
plt.savefig('LossChart.png')
plt.close()

# 绘制准确度折线图
plt.plot(accuriacies)
plt.scatter(range(len(accuriacies)), accuriacies, s=30, marker='o')
plt.legend(['accuracy'])
plt.savefig('AccuracyChart.png')
plt.close()
