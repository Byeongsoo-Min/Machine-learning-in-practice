import torch
from torchvision import datasets, models, transforms
import PIL
from network.inception_resnet_v1 import InceptionResnetV1
from network.TorchUtils import TorchModel


data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

img = PIL.Image.open("real/Me4.jpg")
img_t = data_transforms(img)
img_t = img_t[None,:,:,:]
print(img_t.shape)
model = TorchModel.load_model("exps/models/epoch_50.pt")
print(model)
label = model(img_t)
max_value = 0
max_index = 0
print(label)
for i, value in enumerate(label[0]):
    if max_value < value:
        max_index = i
        max_value = value

print(max_index)
print(max_value)