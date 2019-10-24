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
IMAGE_NAME_FILE = 'top_600.txt'
TRAINED_MODEL_DIR = 'ssd.pytorch/weights/'
TRAINED_MODEL_FN = 'ssd300_mAP_77.43_v2.pth'
TRAINED_MODEL_PATH = path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_FN)

def parse_args():
    parser = argparse.ArgumentParser(description='Measure FPS of Single Shot MultiBox Detector')

    parser.add_argument('--trained-model', required=False, help='Path to trained state_dict file', 
                        default=TRAINED_MODEL_PATH)
    parser.add_argument('--num-classes', required=False, type=int, help='number of classes', 
                        default=21)
    parser.add_argument('--image-path', required=True, help='path to COCO val2014 images dir')
    parser.add_argument('--image-name-file', required=False, help='path to image name file', 
                        default=IMAGE_NAME_FILE)
    parser.add_argument('--image-size', required=False, type=int)
    parser.add_argument('--num-tests', required=False, type=int, default=10)
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

    return {'avg_per_image_ms': avg_sec * 1000, 'avg_per_image_s': avg_sec, 
            'avg_fps': 1 / avg_sec, 'points': list(map(lambda t: t[0], ps))}

def average_averages(args, phase, size, times, tm_func, ik):
    totals = {}
    for t in range(0, times):
        out_m = tm_func(args=args, phase=phase, size=size)
        totals = {k: totals[k] + out_m[k] if k in totals else 0.0\
                  for k in set(out_m).difference(ik)}

    return {k: totals[k] / times for k in totals}

def set_default_tensor_type():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("ERROR: cuda is not available. Test will exit.")

def print_output(args, out_data):
    print("\n\nFrames Per Second result for SSD, averaged over %d runs:" % (args.num_tests))
    for k, v in out_data.items():
        print("%s = %f" % (str(k).ljust(25), v))
    print("\n\n")

if __name__ == '__main__':
    set_default_tensor_type()
    args = parse_args()

    avgs = average_averages(args=args, phase=PHASE, size=IMAGE_SIZE, 
                            times=args.num_tests, tm_func=test_model, ik={'points'})

    print_output(args=args, out_data=avgs)
 
