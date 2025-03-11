import torchvision.transforms as transforms
import torchvision
import torch
import matplotlib.pyplot as plt

post_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((8, 8)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True)


def hash_image(image: torch.Tensor):
    hash = 0
    for i in range(64):
        if image[i] > 0.5:
            hash |= (1 << i)

    return hash


def hamming_distance(hash1: int, hash2: int):
    return bin(hash1 ^ hash2).count('1')


hashs = []

for i, item in enumerate(dataset):
    if i == 10000:
        break
    hashs.append(hash_image(post_transform(item[0]).flatten()))

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
axs[0].imshow(dataset[min_pair[0]][0])
axs[1].imshow(dataset[min_pair[1]][0])
plt.savefig("ahash_result.png")
plt.close()
