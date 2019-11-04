import torch
import argparse
import os.path as path
import sys
import json
import os
import shutil
import importlib
import net_funcs
import time
from torch.autograd import Variable
import numpy as np

MODEL_PATH = './CornerNet'
sys.path.append(MODEL_PATH)
from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory

sys.path.append('..')
import common as com

NET_NAME = 'CornerNet512'
IMAGE_SIZE = 512
SPLIT = 'testing'
DB = 'db'
TRAINED_MODEL_DIR = './weights'
TRAINED_MODEL_FN = 'CornerNet_500000.pkl'
TRAINED_MODEL_PATH = path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_FN)
TEST_DEV_ANNOS = './data/coco/annotations/'
TEST_DEV_ANNOS_FN = 'instances_testdev2017.json'
TEST_DEV_PATH = path.join(TEST_DEV_ANNOS, TEST_DEV_ANNOS_FN)
CACHE_PATH = './cache/nnet/CornerNet'

def parse_args():
    parser = com.default_args(net_name=NET_NAME, trained_model_path=TRAINED_MODEL_PATH, 
                              num_classes=81, image_size=IMAGE_SIZE)
    parser.add_argument('--model-config', required=False, default='CornerNet')
    parser.add_argument('--suffix', required=False, default='.json')
    return parser.parse_args()

def build_model(dataset):
    corner_net = NetworkFactory(dataset)
    corner_net.load_params(system_configs.max_iter)
    corner_net.cuda()
    corner_net.eval_mode()

    return corner_net   

def time_model(model, data, dataset):
    start = time.time()
    out = net_funcs.inference(dataset=dataset, nnet=model, image=data)
    end = time.time()
    
    return end - start, out

def average_inference(model, im_data, dataset, tm_func=time_model):
    points = []
    for i, img in enumerate(im_data):
        if i % 50 == 0:
            print("on image: %d" % (i + 1))
        points.append(tm_func(model=model, dataset=dataset, data=img))

    return points, np.average(list(map(lambda t: t[0], points)))

def test_model(args, size, model, dataset):
    images, nf = com.load_images(args.image_path, 
                         com.process_file_names(com.read_file(args.image_name_file)))
    resized_images = com.resize_images(size=size, images=images)
    return com.test_model(inference_func=average_inference, 
                          inference_func_args={'model': model, 'im_data': resized_images, 
                                               'dataset': dataset})

def average_averages(args, size, model, dataset, times, ik):
    return com.average_averages(times=times, ik=ik, tm_func=test_model, 
                                tm_args={'args': args, 'size': size, 'model': model, 
                                         'dataset': dataset})

def prepare_dirs():
    com.make_dirs([TEST_DEV_ANNOS, CACHE_PATH])
    shutil.copy(path.join('./', TEST_DEV_ANNOS_FN), TEST_DEV_PATH)
    shutil.copy(TRAINED_MODEL_PATH, CACHE_PATH)

def create_config(args):
    model_config_path = args.model_config + args.suffix
    config_fp = path.join(MODEL_PATH, path.join(system_configs.config_dir, model_config_path))
    config = com.read_json_file(path=config_fp)
    config['system']['snapshot_name'] = args.model_config
    system_configs.update_config(config['system'])
    test_dataset = datasets[system_configs.dataset](config[DB], system_configs.test_split)

    return config, test_dataset

if __name__ == '__main__':
    args = parse_args()
    prepare_dirs()
    config, test_dataset = create_config(args=args)
    net = build_model(dataset=test_dataset)
    out = average_averages(args=args, size=IMAGE_SIZE, model=net, dataset=test_dataset,
                           times=args.num_tests, ik={'points'})
    com.print_output(args=args, out_data=out, model_name=NET_NAME) 

    
