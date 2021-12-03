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
class_list = os.listdir(faces_path)
all_file_lst = []
for face_class in class_list:
    file_list = os.listdir(os.path.join(faces_path , face_class))
    all_file_lst.append((file_list[0], os.path.join(faces_path, face_class, file_list[0])))



i = 0
for file, path in all_file_lst:
    os.rename(path, dest_path + "/{}.jpg".format(i))
    i += 1
"""
