import torch
import numpy as np
from pytorch_grad_cam import GradCAM, GuidedBackpropReLU
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from models import SCCModel

def visualize_xai(model, test_loader):
    model.eval()
    data, _ = next(iter(test_loader))
    data = data.to(model.device).float()
    target = torch.tensor([0]).to(model.device)  # SCC class

    # Grad-CAM
    target_layers = [list(model.model.children())[-2] if 'inception' in model.model.__class__.__name__ else list(model.model.children())[-3]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=data[0].unsqueeze(0), targets=None)
    cam_image = show_cam_on_image(data[0].cpu().numpy().transpose((1, 2, 0)), grayscale_cam[0], use_rgb=True)

    # Guided Backprop
    gbp = GuidedBackpropReLU(model)
    guided_grads = gbp(data[0].unsqueeze(0), target)
    guided_grads = np.transpose(guided_grads.cpu().numpy(), (1, 2, 0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cam_image)
    plt.title('Grad-CAM')
    plt.subplot(1, 2, 2)
    plt.imshow(guided_grads)
    plt.title('Guided Backprop')
    plt.show()