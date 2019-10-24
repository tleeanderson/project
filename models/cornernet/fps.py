import argparse
import os.path as path
import sys

MODEL = 'CornerNet'
MODEL_PATH = path.join('./', MODEL)
sys.path.append(path.abspath(MODEL_PATH))
from config import system_configs

PHASE = 'test'
IMAGE_SIZE = 300
IMAGE_NAME_FILE = 'top_600.txt'
TRAINED_MODEL_DIR = 'cornernet/weights/'
TRAINED_MODEL_FN = 'CornerNet_500000.pkl'
TRAINED_MODEL_PATH = path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_FN)

def parse_args():
    parser = argparse.ArgumentParser(description='Measure FPS of CornerNet')

    parser.add_argument('--trained-model', required=False, help='Path to trained state_dict file', 
                        default=TRAINED_MODEL_PATH)
    parser.add_argument('--num-classes', required=False, type=int, help='number of classes', 
                        default=81)
    parser.add_argument('--image-path', required=True, help='path to COCO val2014 images dir')
    parser.add_argument('--image-name-file', required=False, help='path to image name file', 
                        default=IMAGE_NAME_FILE)
    parser.add_argument('--image-size', required=False, type=int)
    parser.add_argument('--num-tests', required=False, type=int, default=10)
    parser.add_argument('--model-config', required=False, default='CornerNet.json')
    return parser.parse_args()

def read_file(path):
    with open(path) as f:
        data = f.read()

    return data



if __name__ == '__main__':
    args = parse_args()
    
    config_fp = path.join(MODEL_PATH, path.join(system_configs.config_dir, args.model_config))
    config_file = read_file(config_fp)

    print(config_file)
