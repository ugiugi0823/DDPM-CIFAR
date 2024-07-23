import torch
from torchvision.utils import save_image
from ddpm_cond_pl import ConditionalDiffusionModel, Diffusion

class Args:
    def __init__(self):
        self.num_classes = 10
        self.image_size = 32
        # 필요한 다른 속성들도 여기에 추가하세요

n = 1
device = "cuda"
args = Args()  # args 객체 생성

ckpt = torch.load("/media/hy/nwxxk/NeurIPS/DDPM-CIFAR/checkpoints/2024-07-22T17-30-09/epoch=56-val_loss=0.02.ckpt")
model = ConditionalDiffusionModel(args).to(device)
model.load_state_dict(ckpt['state_dict'])

# 이미지 생성
diffusion = Diffusion(img_size=args.image_size, device=device)
y = torch.Tensor([6] * n).long().to(device)
x = diffusion.sample(model, n, y, cfg_scale=3)

# 텐서의 값 범위 확인
print(f"Tensor min: {x.min()}, max: {x.max()}")

# 텐서를 0-1 범위로 정규화하고 이미지로 저장
x_normalized = x.float() / 255.0  # 0-255 범위를 0-1 범위로 변환
save_image(x_normalized, "generated_image2.png", normalize=False)

print("Image saved as 'generated_image.png'")