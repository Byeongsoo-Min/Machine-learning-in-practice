import torch
from torchvision import datasets, models, transforms
import PIL
import os
import numpy as np
from IQA_pytorch import SSIM, utils
import cv2
import argparse

from network.TorchUtils import TorchModel
from face_point import face_points

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIV6 Face Parser")

    # io
    parser.add_argument('--input_path', help="path to input")
    parser.add_argument('--out_video_name', help="output filename")
    return parser.parse_args()

def generate_demo(img_path):

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    data_transforms_nomal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((500, 500)),
        ])

    demo_img = img_path

    img = PIL.Image.open(demo_img)
    img_t = data_transforms(img)
    img_t = img_t[None,:,:,:]
    model = TorchModel.load_model("C:/deep/Machine-learning-in-practice/exps/models/epoch_55.pt")
    model.eval()
    label = model(img_t).detach().numpy()
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
    class_face_path = "C:/deep/Machine-learning-in-practice/videos/faces/{}".format(class_name)
    filelist = os.listdir(class_face_path)
    # class_tensors = []
    # class_tensors_path = "class_tensors/"+class_name+".npy"
    # if not os.path.exists(class_tensors_path):
    #     for file in filelist:
    #         img = PIL.Image.open(os.path.join(class_face_path, file))
    #         img_t_ = data_transforms(img)
    #         class_tensors.append(np.array(img_t_))
    #     np.save(class_tensors_path, class_tensors)

    # class_tensors = np.load(class_tensors_path, allow_pickle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSIM(channels=3)
    img_t = utils.prepare_image(np.array(data_transforms_nomal(PIL.Image.open(demo_img).convert("RGB")))).to(device)
    img_t = img_t.swapaxes(1,2)
    scores = []

    for file in filelist:
        tensor = utils.prepare_image(PIL.Image.open(os.path.join(class_face_path, file)).convert("RGB")).to(device)
        # print(img_t.shape, tensor.shape)
        score = model(img_t, tensor, as_loss=False)
        scores.append(score)
    max_index = scores.index(max(scores))
    dest_img = filelist[max_index]
    img = cv2.imread(os.path.join(class_face_path, dest_img))
    while len(face_points(img)) == 0:
        scores[max_index] = 0
        max_index = scores.index(max(scores))
        dest_img = filelist[max_index]
        img = cv2.imread(os.path.join(class_face_path, dest_img))

    src = demo_img
    dest = class_face_path +"/" + dest_img

    os.system("python C:\\deep\\face_morpher\\facemorpher\\morpher.py --src={} --dest={} --out_video={}".format(src, dest, args.out_video_name))


if __name__ == "__main__":
    args = get_args()
    generate_demo(args.input_path)