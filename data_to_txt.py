import argparse
import math
import os
import random
import numpy as np

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


def yazdir(class_id, path, dir, image_id, folder):
    # print(path)
    files = os.listdir(path)
    for item in files:
        file = path + "/" + item
        if os.path.isfile(file):
            image_id += 1
            file_path = dir + "/" + item
            folder.write(str(image_id) + " " + str(class_id) + " " + "1" + " " + str(file_path.replace('\\', '/')) + "\n")

    return image_id


all = os.listdir(args.path)
random.shuffle(all)
folders = np.array([folder for folder in all if os.path.isdir(os.path.join(args.path, folder))])

break_point1 = int(math.ceil(len(folders) / 2))
break_point2 = int(math.ceil(break_point1 / 2))

trainval = folders[:break_point1]
eval = folders[break_point1:]
train = trainval[:break_point2]
val = trainval[break_point2:]

print('"train": "range(0, %d)", \n "val": "range(%d, %d)", \n "trainval": "range(0, %d)", \n "eval": "range(%d, %d)"' % (break_point2, break_point2, break_point1, break_point1, break_point1, len(folders)))

# klasörleri listeleme
for item in trainval:
    if os.path.isdir(args.path + "/" + item):
        class_id += 1
        p = (args.path + "/" + item)
        image_id = yazdir(class_id, p, item, image_id, f_train)


print("nb_train_all: %d" % image_id)
temp_image_id = image_id


for item in eval:
    if os.path.isdir(args.path + "/" + item):
        class_id += 1
        p = (args.path + "/" + item)
        image_id = yazdir(class_id, p, item, image_id, f_test)

print("nb_test_all: %d" % (image_id - temp_image_id))

