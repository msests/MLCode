import torch
import matplotlib.pyplot as plt


def func(args):
    return args[0]**2 + 10*args[1]*3 + args[2]


def gen_training_set():
    y_hat = torch.zeros(1000)
    # 生成1000*3个-100到100之间均匀分布的随机数
    x = torch.rand(1000, 3) * 10 - 5
    for i in range(1000):
        y_hat[i] = func(x[i]) + torch.randn(1) * 0.1
    return x, y_hat


class MLPModel(torch.nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)


if torch.cuda.is_available():
    print('CUDA is available, device:', torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = MLPModel().to(device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

x, y_hat = gen_training_set()
x = x.to(device)
y_hat = y_hat.to(device)

print('training...')

for epoch in range(1000):
    for i in range(1000):
        y_pred = net(x[i])
        y_pred = y_pred.squeeze().to(device)
        loss_value = loss(y_pred, y_hat[i])
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss_value.item()}')
        # print(f'epoch: {epoch}, loss: {loss_value.item()}')
