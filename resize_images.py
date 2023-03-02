import os
import numpy as np
import cv2
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)


def read_image(*args):
    img = cv2.imread(join_dir(*args), cv2.IMREAD_COLOR)
    return img


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def save_image(img, img_name):
    cv2.imwrite(img_name, img)


def resize_Image(img):
    img = cv2.resize(img, (100, 56), interpolation = cv2.INTER_AREA)
    return img


def list_files(directory):
    files = os.listdir(join_dir(cwd, directory))
    files = filter(lambda x: not x.startswith('.'), files)
    return list(files)


def resize_images(dir_1, dir_2, threshold_value):
    g_t = list_files(join_dir('Dataset',dir_1))
    count = 1
    for i in (sorted(g_t)):
        # Mention the directory in which you wanna resize the images followed by the image name
        img = read_image("Dataset", dir_1, i)
        # reading the more emphasis blue channel
        img_b = img[:, :, 2]
        res, th2 = cv2.threshold(np.array(img_b), threshold_value, 255, cv2.THRESH_BINARY_INV)
        img = resize_Image(th2)
        save_image(img, join_dir(cwd, 'resized', dir_2,f'{dir_2}_{count}.png'))
        count = count + 1


desc = "Resize images"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--images_original', type=str,
                    help='Specify directory of your dataset images  ')

parser.add_argument('-r', '--images_resized', type=str,
                    help='Specify directory to save the resized images  ')

parser.add_argument('-t', '--threshold', type=int, default=30,
                    help='Specify threshold')
args = parser.parse_args()

if args.images_original and args.images_resized:
    check_dir(join_dir(cwd, 'resized', args.images_resized))
    resize_images(args.images_original, args.images_resized, args.threshold)
else:
    exit()



