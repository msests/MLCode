import torch
import matplotlib.pyplot as plt


def func(args):
    return args[0]*0.4 + args[1]*10 + args[2]*3


def gen_train_set():
    y_hat = torch.zeros(1000)
    # 生成1000*3个-5到5之间均匀分布的随机数
    x = torch.rand(1000, 3) * 10 - 5
    for i in range(1000):
        y_hat[i] = func(x[i]) + torch.randn(1) * 0.1
    return x, y_hat


def gen_test_set():
    y = torch.zeros(100)
    x = torch.rand(100, 3) * 10 - 5
    for i in range(100):
        y[i] = func(x[i])
    return x, y

# 定义线性回归网络


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 1)
        )

    def forward(self, x):
        return self.net(x)


if torch.cuda.is_available():
    print('CUDA is available, device:', torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化网络
net = LinearRegression().to(device)

# 使用均方误差损失函数
loss = torch.nn.MSELoss()

# 使用随机梯度下降优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

x, y_hat = gen_train_set()
x = x.to(device)
y_hat = y_hat.to(device)


print('training...')

for i in range(1000):
    y_pred = net(x[i])
    y_pred = y_pred.squeeze().to(device)
    loss_value = loss(y_pred, y_hat[i])
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f'loss: {loss_value.item()}')

# 测试网络准确率

print('testing...')

net.eval()

test_x, test_y = gen_test_set()
test_x = test_x.to(device)
test_y = test_y.to(device)

errs = torch.zeros(test_x.shape[0]).to(device)
err_mean = torch.zeros(1).to(device)

for i in range(test_x.shape[0]):
    y_pred = net(test_x[i])
    y_pred = y_pred.squeeze().to(device)
    errs[i] = torch.abs(y_pred - test_y[i])

err_mean = torch.mean(errs)
print(f'average error: {err_mean.item()}')
