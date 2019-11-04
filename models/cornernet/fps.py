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
import common

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
    parser = common.default_args(net_name=NET_NAME, trained_model_path=TRAINED_MODEL_PATH, 
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

def inference(model, dataset, image):
    return common.time_inference(inference_func=net_funcs.inference, 
                              inference_func_args={'dataset': dataset, 
                                                   'nnet': model, 
                                                   'image': image})

def test_model(args, size, model, dataset):
    images = common.read_images(image_name_file=args.image_name_file, 
                                image_path=args.image_path, size=size)
    return common.test_model(im_data=images, inference_func=inference,
                             inference_func_args={'model': model,
                                                   'dataset': dataset})

def average_averages(args, size, model, dataset, times, ik):
    return common.average_averages(times=times, ik=ik, test_model_func=test_model, 
                                test_model_args={'args': args, 'size': size, 'model': model, 
                                         'dataset': dataset})

def prepare_dirs():
    common.make_dirs([TEST_DEV_ANNOS, CACHE_PATH])
    shutil.copy(path.join('./', TEST_DEV_ANNOS_FN), TEST_DEV_PATH)
    shutil.copy(TRAINED_MODEL_PATH, CACHE_PATH)

def create_config(args):
    model_config_path = args.model_config + args.suffix
    config_fp = path.join(MODEL_PATH, path.join(system_configs.config_dir, model_config_path))
    config = common.read_json_file(path=config_fp)
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
    common.print_output(args=args, out_data=out, model_name=NET_NAME) 

    
