import argparse
import math
import os
import numpy as np

from pathlib import Path

parser = argparse.ArgumentParser(description='Train Data info')
parser.add_argument('--path', default='example')

args = parser.parse_args()

f_train = open(args.path + "/Shoes_train.txt", "w+")
f_test = open(args.path + "/Shoes_test.txt", "w+")
f_train.write("image_id class_id super_class_id path" + "\n")
f_test.write("image_id class_id super_class_id path" + "\n")

image_id = 0
class_id = 0
counts = []

data_path = Path(args.path)
folders = np.array([folder for folder in data_path.iterdir() if folder.is_dir() and len(list(folder.iterdir())) > 2])
np.random.shuffle(folders)

break_point_half = int(math.ceil(len(folders)))
break_point_quarter = int(math.ceil(break_point_half / 2))
break_point3 = int(math.ceil(break_point_quarter / 2))

trainval = folders[:break_point_quarter]
eval = folders[break_point_quarter:break_point_half]
train = trainval[:break_point3]
val = trainval[break_point3:break_point_quarter]

print(
    '"train": "range(0, %d)", \n "val": "range(%d, %d)", \n "trainval": "range(0, %d)", \n "eval": "range(%d, %d)"' % (
    break_point3, break_point3, break_point_quarter, break_point_quarter, break_point_quarter, break_point_half))


def yazdir(class_id, item, image_id, folder):
    for file in item.iterdir():
        image_id += 1
        file_path = os.path.join(item.name, file.name)
        folder.write(
            str(image_id) + " " + str(class_id) + " " + "1" + " " + file_path + "\n")

    return image_id


# klasörleri listeleme
for item in trainval:
    class_id += 1
    image_id = yazdir(class_id, item, image_id, f_train)

print("nb_train_all: %d" % image_id)
temp_image_id = image_id

for item in eval:
    class_id += 1
    image_id = yazdir(class_id, item, image_id, f_test)

print("nb_test_all: %d" % (image_id - temp_image_id))
