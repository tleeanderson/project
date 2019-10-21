import time
import argparse
import torch
import cv2
import os.path as path
import sys
from torch.autograd import Variable
sys.path.append(path.abspath('./ssd.pytorch'))

import ssd as model
import numpy as np
import datetime

PHASE = 'test'
IMAGE_SIZE = 300

def parse_args():
    parser = argparse.ArgumentParser(description='Measure Single Shot MultiBox Detector FPS')

    parser.add_argument('--trained-model', required=False, help='Path to trained state_dict file')
    parser.add_argument('--num-classes', required=True, type=int, help='number of classes')
    parser.add_argument('--image-path', required=True, help='path to images')
    parser.add_argument('--image-name-file', required=True, help='path to image name file')
    parser.add_argument('--image-size', required=False, type=int)
    return parser.parse_args()

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

def build_model(args, phase):
    ssd_model = model.build_ssd(phase, 300, args.num_classes)
    ssd_model.load_state_dict(torch.load(args.trained_model))
    ssd_model.eval()
    ssd_model = ssd_model.cuda()

    return ssd_model

def time_model(model, data):
    test_data = Variable(data)
    test_data = test_data.cuda()
    start = time.time()
    out = model(test_data)
    end = time.time()
    
    return end - start, out

def average_inference(model, im_data):
    points = [time_model(model, i) for i in im_data]
    return points, np.average(list(map(lambda t: t[0], points)))

def test_model(args, phase, size):
    images, nf = load_images(args.image_path, 
                             process_file_names(read_file(args.image_name_file)))
    resized_images = prepare_images(resize_images(size, images), size)

    ssd_model = build_model(args, phase)
    ps, avg_sec = average_inference(ssd_model, resized_images)

    print("Average per image(ms): %f, Average per image(s): %f, Average FPS: %f" 
          % (avg_sec * 1000, avg_sec, 1 / avg_sec))
    print("ps: " + str(list(map(lambda t: t[0], ps))))

def set_default_tensor_type():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("ERROR: cuda is not available. Test will exit.")

if __name__ == '__main__':
    set_default_tensor_type()
    args = parse_args()
    test_model(args, PHASE, int(args.image_size) if args.image_size else IMAGE_SIZE)
 
