import torch
from torchvision.utils import save_image
from ddpm_cond_pl import Diffusion
from modules import UNet_conditional




for i in range(10):
    num_img = 1 # 몇개 생성
    class_num = 2 # 생성할 클래스 번호
    device = "cuda"
    ckpt = torch.load("/media/hy/nwxxk/NeurIPS/DDPM-CIFAR/wxxk_conditional_ema.pt")
    model = UNet_conditional(num_classes=10).to(device)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=64, device=device)
    y = torch.Tensor([class_num] * num_img).long().to(device)
    x = diffusion.sample(model, num_img, y, cfg_scale=3)
        
    # 생성된 이미지 저장
    # 텐서를 0-1 범위로 정규화하고 이미지로 저장
    x_normalized = x.float() / 255.0  # 0-255 범위를 0-1 범위로 변환
    save_image(x_normalized, f"./wxxk/generated_image{i}.png", normalize=False)



"""
CIFAR-10 데이터셋의 클래스 개수는 10개입니다. CIFAR-10은 10개의 클래스
    0: airplane, 
    1: automobile, 
    2: bird, 
    3: cat, 
    4: deer, 
    5: dog, 
    6: frog, 
    7: horse, 
    8: ship, 
    9: truck
"""