from vision_transformer import *
from torchinfo import summary
import os
import torch
model = vit_tiny_patch16_224(pretrained=True)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output['pre_logits'].shape)
print("Model created successfully.")
