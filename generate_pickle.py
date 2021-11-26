
from facenet_pytorch import InceptionResnetV1
import torch

model = InceptionResnetV1(pretrained='casia-webface')
state_dict = model.state_dict()

keys_to_remove = []
for key in model.state_dict().keys():
    if 'last_bn' in key:
        keys_to_remove.append(key)
    if 'logits' in key:
        keys_to_remove.append(key)

print(keys_to_remove)
for key in keys_to_remove:
    state_dict.pop(key)

torch.save(state_dict, "pretrained.pth")