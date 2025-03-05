import torchvision.transforms as transforms
import torchvision
import torch
import matplotlib.pyplot as plt

post_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((8, 9)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_diff_hash(image: torch.Tensor):
    hash = 0
    for i in range(8):
        for j in range(8):
            if image[i][j] > image[i][j + 1]:
                hash |= (1 << (i * 8 + j))

    print(hash)

    return hash


def image_diff_hash_opt(image: torch.Tensor):
    hash = 0

    image = image.to(device)

    left = image[:, :-1]
    right = image[:, 1:]
    diff = left - right

    hash_cpu = diff.flatten().cpu()
    for i in range(64):
        if hash_cpu[i] > 0:
            hash |= (1 << i)

    return hash


def hamming_distance(hash1: int, hash2: int):
    return bin(hash1 ^ hash2).count('1')


hashs = []

for i, item in enumerate(dataset):
    if i == 1000:
        break
    hashs.append(image_diff_hash_opt(post_transform(item[0]).squeeze(0)))

min_dis = 2**32 - 1
min_pair = None

for i in range(1000):
    for j in range(i + 1, 1000):
        dis = hamming_distance(hashs[i], hashs[j])
        if dis < min_dis:
            min_dis = dis
            min_pair = (i, j)

# 用matplotlib并排显示min_pair的两张图片
fig, axs = plt.subplots(1, 2)
axs[0].imshow(dataset[min_pair[0]][0])
axs[1].imshow(dataset[min_pair[1]][0])
plt.show()
