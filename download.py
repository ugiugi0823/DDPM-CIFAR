import os
from torchvision import datasets, transforms

# 데이터 저장 경로 지정
base_path = '/media/hy/nwxxk/NeurIPS/Diffusion-Models-pytorch-my/datasets'
mnist_path = os.path.join(base_path, 'MNIST')
cifar_path = os.path.join(base_path, 'CIFAR10')

# 경로가 존재하지 않으면 생성
os.makedirs(mnist_path, exist_ok=True)
os.makedirs(cifar_path, exist_ok=True)

# MNIST 데이터 다운로드 및 저장
mnist_train = datasets.MNIST(root=mnist_path, train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root=mnist_path, train=False, download=True, transform=transforms.ToTensor())

# CIFAR-10 데이터 다운로드 및 저장
cifar_train = datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR10(root=cifar_path, train=False, download=True, transform=transforms.ToTensor())

print("MNIST and CIFAR-10 datasets have been downloaded and saved.")
