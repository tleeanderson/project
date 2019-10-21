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

PHASE = 'test'
IMAGE_SIZE = 300

def parse_args():
    parser = argparse.ArgumentParser(description='Measure Single Shot MultiBox Detector FPS')

    parser.add_argument('--trained-model', required=False, help='Path to trained state_dict file')
    parser.add_argument('--num-classes', required=True, type=int, help='number of classes')
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--image-names', required=True, nargs='+')
    parser.add_argument('--image-size', required=False)
    return parser.parse_args()

def average_inference(model):
    pass

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
    gpu_image = data.cuda()

    # i = 10
    # while i > 0:
    #     print("Waiting %d seconds to continue" % (i))
    #     i -= 1
    #     time.sleep(1)

    start = time.time()
    out = model(gpu_image).data
    end = time.time()
    
    return end - start, out

def average_inference(model, im_data):
    points = [time_model(model, i) for i in im_data]
    return points, np.average(list(map(lambda t: t[0], points)))

def test_model(args, phase, size):
    images, nf = load_images(args.image_path, args.image_names)
    resized_images = prepare_images(resize_images(size, images), size)

    ssd_model = build_model(args, phase)
    ps, avg_ms = average_inference(ssd_model, resized_images)

    print("points: %s, avg(ms): %d" % (str(list(map(lambda t: t[0], ps))), avg_ms))
    
if __name__ == '__main__':
    args = parse_args()
    test_model(args, PHASE, IMAGE_SIZE)
 
