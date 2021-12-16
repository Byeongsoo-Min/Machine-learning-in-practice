import shutil
import os

from_path = 'face2civ/B'
dest_path = 'face2civ'


file_list = os.listdir(from_path)
for file in file_list:
    if int(file.split(".")[0]) % 2 == 1:
        os.rename(os.path.join(from_path, file), dest_path + "/trainB/" + file)
    else:
        os.rename(os.path.join(from_path, file), dest_path + "/testB/" + file)



"""

faces_path = "C:/deep/Machine-learning-in-practice/videos/civ_face"
dest_path = "C:/deep/pytorch-CycleGAN-and-pix2pix/datasets/face2civ_v2/A"

class_list = os.listdir(faces_path)
all_file_lst = []
for face_class in class_list:
    file_list = os.listdir(os.path.join(faces_path , face_class))
    for face in file_list:
        all_file_lst.append(os.path.join(faces_path, face_class, face))


i = 0
for path in all_file_lst:
    os.rename(path, dest_path + "/{}.jpg".format(i))
    i += 1
"""