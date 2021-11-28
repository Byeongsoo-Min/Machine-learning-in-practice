import torch
from torchvision import datasets, models, transforms
import PIL
from network.inception_resnet_v1 import InceptionResnetV1
from network.TorchUtils import TorchModel
import os
import random


data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

demo_img = "real\\young.jpg"

img = PIL.Image.open(demo_img)
img_t = data_transforms(img)
img_t = img_t[None,:,:,:]
#print(img_t)
model = TorchModel.load_model("exps/models/epoch_55.pt")
model.eval()
label = model(img_t)
max_value = 0
max_index = 0

for i, value in enumerate(label[0]):
    value = float(value)
    if max_value < value:
        max_index = i
        max_value = value
#classes = ['Chandragupta_faces', 'Cleopatra_faces', 'Cyrus_faces', 'Gandhi_faces', 'Genghis_Khan_faces', 'Gitarja_faces', 'Hammurabi_faces', 'Hojo_Tokimune_faces', 'Jadwiga _faces', 'Jayavarman_VII_faces', 'Kristina_faces', 'Kupe_faces', 'MansaMusa_faces', 'Mvemba_faces', 'Pachacuti_faces', 'Pedro_faces', 'Peter_faces', 'Philip_II_faces', 'QinShiHuang_faces', 'Robert_faces', 'Seondeok_faces', 'Simón_Bolívar_faces', 'Suleiman_faces', 'Teddy_Roosevelt_faces', 'Trajan_faces', 'Victoria_faces', 'catherine_faces', 'dido_faces']
classes = ['Alexander_faces', 'Amanitore_faces', 'Ambiorix_faces', 'Ba_Trieu_faces', 'Basil_II_faces', 'Catherine_faces', 'Chandragupta_faces', 'Cleopatra_faces', 'Cyrus_faces', 'Dido_faces', 'Eleanor_of_Aquitaine_faces', 'Frederick_Barbarossa_faces', 'Gandhi_faces', 'Genghis_Khan_faces', 'Gilgamesh_faces', 'Gitarja_faces', 'Gorgo_faces', 'Hammurabi_faces', 'Hojo_Tokimune_faces', 'Jadwiga_faces', 'Jayavarman_VII_faces', 'Joao_III_faces', 'John_Curtin_faces', 'Kristina_faces', 'Kublai_Khan_faces', 'Kupe_faces', 'MansaMusa_faces', 'Matthias_Corvinus_faces', 'Menelik_II_faces', 'Mvemba_faces', 'Pachacuti_faces', 'Pedro_faces', 'Peter_faces', 'Philip_II_faces', 'Poundmaker_faces', 'QinShiHuang_faces', 'Robert_faces', 'Seondeok_faces', 'Shaka_faces', 'Simón_Bolívar_faces', 'Suleiman_faces', 'Tamar_faces', 'Teddy_Roosevelt_faces', 'Tomyris_faces', 'Trajan_faces', 'Victoria_faces', 'Wilfrid_Laurier_faces', 'Wilhelmina_faces']
class_name = classes[max_index]
print(class_name)
print(max_value)
filelist = os.listdir("videos/faces/{}".format(class_name))
dest_img = random.choice(filelist)
current_dir = os.getcwd()
src = current_dir+"\\{}".format(demo_img)
dest = current_dir+"\\videos\\faces\\{}\\{}".format(class_name, dest_img)

os.system("python C:\\deep\\face_morpher\\facemorpher\\morpher.py --src={} --dest={} --out_video=out.avi".format(src, dest))


