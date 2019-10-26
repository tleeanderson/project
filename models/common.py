import os.path as path
import cv2
import torch

def read_file(path):
    with open(path) as f:
        data = f.read()

    return data

def process_file_names(data):
    return data.split('\n')[:-1]

def resize_images(size, images):
    return [cv2.resize(i, (size, size)) for i in images]

def prepare_images(images, size):
    return [torch.from_numpy(i).unsqueeze(0)\
            .reshape((1, 3, size, size)).float() for i in images]

def load_images(im_path, names):
    images, nf = [], []
    for n in names:
        p = path.join(im_path, n)
        im = cv2.imread(p)
        if im is None:
            nf.append(p)
        else:
            images.append(im)
    return images, nf
