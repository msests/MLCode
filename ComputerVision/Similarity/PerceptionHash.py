import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

post_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=post_transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dct_matrix(N: int = 32):
    n = torch.arange(N).float()
    k = torch.arange(N).float().reshape(-1, 1)

    # [32*1] * [1*32] = [32*32]
    dct_base_matrix = torch.cos(torch.pi * k * (2*n+1) / (2*N))

    dct_base_matrix[0, :] *= 1 / torch.sqrt(torch.tensor(2.0))
    dct_base_matrix *= torch.sqrt(torch.tensor(2.0) / torch.tensor(N))

    return dct_base_matrix


dct_base_matrix = get_dct_matrix()


def dct_2d(image: torch.Tensor):
    return torch.matmul(torch.matmul(dct_base_matrix, image), dct_base_matrix.T)


print(dataset[0][0].shape)


def image_p_hash(image: torch.Tensor):
    dct_mat = dct_2d(image)
    top_lf = dct_mat[0:8, 0:8].flatten()
    for i in range(64):
        if top_lf[i] > top_lf.mean():
            top_lf[i] = 1
        else:
            top_lf[i] = 0

    hash = 0
    for i in range(8):
        for j in range(8):
            if top_lf[i * 8 + j] == 1:
                hash |= (1 << (i * 8 + j))

    return hash


def hamming_distance(hash1: int, hash2: int):
    return bin(hash1 ^ hash2).count('1')


X = dct_2d(dataset[0][0])

print(X.shape)

hashs = []

for i, item in enumerate(dataset):
    if i == 10000:
        break
    hashs.append(image_p_hash(item[0]))

min_dis = 2**32 - 1
min_pair = None

for i in range(10000):
    for j in range(i + 1, 10000):
        dis = hamming_distance(hashs[i], hashs[j])
        if dis < min_dis:
            min_dis = dis
            min_pair = (i, j)

# 用matplotlib并排显示min_pair的两张图片
fig, axs = plt.subplots(1, 2)
axs[0].imshow(dataset[min_pair[0]][0][0])
axs[1].imshow(dataset[min_pair[1]][0][0])
plt.savefig("phash_result.png")
plt.close()
