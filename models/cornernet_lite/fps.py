import sys
sys.path.append('..')
import common
from CornerNet_Lite import CornerNet
import time
import cv2

def parse_args():
    parser = common.default_args(net_name=NET_NAME, trained_model_path=TRAINED_MODEL_PATH, 
                              num_classes=81, image_size=IMAGE_SIZE)
    parser.add_argument('--model-config', required=False, default='CornerNet')
    parser.add_argument('--suffix', required=False, default='.json')
    return parser.parse_args()

if __name__ == '__main__':
    img = cv2.imread('/home/tanderson/models/datasets/mscoco/raw-data/val2014/COCO_val2014_000000581632.jpg')
    net = CornerNet()
    start = time.time()
    out = net(img)
    end = time.time()

    print("output time: " + str(end - start))
