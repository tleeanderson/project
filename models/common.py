import os.path as path
import cv2
import torch
import argparse
import json
import time
import numpy as np

IMAGE_NAME_FILE = 'top_600.txt'

def default_args(net_name, trained_model_path, num_classes, image_size, 
                 num_tests=10, image_name_file=IMAGE_NAME_FILE):
    parser = argparse.ArgumentParser(description="Measure FPS of {}".format(net_name))
    parser.add_argument('--trained-model', required=False, help='Path to trained state_dict file', 
                        default=trained_model_path)
    parser.add_argument('--num-classes', required=False, type=int, help='number of classes', 
                        default=num_classes)
    parser.add_argument('--image-path', required=True, help='path to COCO val2014 images dir')
    parser.add_argument('--image-name-file', required=False, help='path to image name file', 
                        default=image_name_file)
    parser.add_argument('--image-size', required=False, type=int, default=image_size)
    parser.add_argument('--num-tests', required=False, type=int, default=num_tests)

    return parser

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

def images_from_disk(im_path, names):
    images, nf = [], []
    for n in names:
        p = path.join(im_path, n)
        im = cv2.imread(p)
        if im is None:
            nf.append(p)
        else:
            images.append(im)
    return images, nf

def time_inference(inference_func, inference_func_args):
    start = time.time()
    out = inference_func(**inference_func_args)
    end = time.time()

    return end - start, out

def average_averages(times, ik, test_model_func, test_model_args):
    totals = {}
    for t in range(1, times + 1):
        print("On test number %d" % (t))
        out_m = test_model_func(**test_model_args)
        totals = {k: totals[k] + out_m[k] if k in totals else out_m[k]\
                  for k in set(out_m).difference(set(ik))}

    return {k: totals[k] / times for k in totals}

def test_model(im_data, inference_func, inference_func_args):
    points = []
    for i, img in enumerate(im_data):
        if i % 50 == 0:
            print("on image: %d" % (i + 1))
        points.append(inference_func(image=img, **inference_func_args))
    avg_sec = np.average(list(map(lambda t: t[0], points)))
    return {'avg_per_image_ms': avg_sec * 1000, 'avg_per_image_s': avg_sec, 
            'avg_fps': 1 / avg_sec, 'points': list(map(lambda t: t[0], points))}

def read_images(image_name_file, image_path, size):
    images, nf = images_from_disk(im_path=image_path, 
                             names=process_file_names(read_file(image_name_file)))
    for p in nf:
        print("ERROR: Could not find {}".format(p))
    resized_images = resize_images(size=size, images=images)
    return resized_images

def print_output(args, out_data, model_name):
    print('out_data ' + str(out_data))
    print("\n\nFrames Per Second result for %s, averaged over %d runs:" 
          % (str(model_name), args.num_tests))
    for k, v in out_data.items():
        print("%s = %f" % (str(k).ljust(25), v))
    print("\n\n")

def read_json_file(path):
    with open(path) as f:
        data = json.load(f)
    
    return data

def make_dirs(dirs):
    list(map(lambda d: os.makedirs(d, exist_ok=True), 
             list(filter(lambda d: not path.exists(d), dirs))))
