# utils.py

import torch
import torch.nn as nn
import torch.autograd as autograd

def weights_init(m):
    """
    Khởi tạo trọng số tùy chỉnh cho các lớp Conv và BatchNorm.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def compute_gradient_penalty(D, real_samples, fake_samples, device, lambda_gp):
    """
    Tính toán gradient penalty loss cho WGAN-GP.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(batch_size, 1, device=device, requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def add_dp_noise_to_gradients(model, noise_multiplier, max_grad_norm):
    """Thêm nhiễu DP vào gradients sau khi đã clip."""
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    
    # Thêm nhiễu Gaussian
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_multiplier * max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad.add_(noise)


                