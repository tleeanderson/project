import sys
sys.path.append('..')
import common
from CornerNet_Lite import CornerNet
from CornerNet_Lite import CornerNet_Saccade
from CornerNet_Lite import CornerNet_Squeeze
import time
import cv2

NET_NAME = 'CornerNet_Lite'
CORNER_NET = 'CornerNet'
CORNER_NET_SQUEEZE = 'CornerNet_Squeeze'
CORNER_NET_SACCADE = 'CornerNet_Saccade'
IMAGE_SIZE = 512

def parse_args():
    parser = common.default_args(net_name=NET_NAME,  num_classes=81, image_size=IMAGE_SIZE)
    parser.add_argument('--model', required=False, default=CORNER_NET, 
                        choices=[CORNER_NET, CORNER_NET_SACCADE, CORNER_NET_SQUEEZE])
    return parser.parse_args()

def inference(model, image):
    return common.time_inference(inference_func=model, 
                                 inference_func_args={'image': image})

def test_model(args, size, model):
    images = common.read_images(image_name_file=args.image_name_file, 
                                image_path=args.image_path, size=size)
    return common.test_model(im_data=images, inference_func=inference, 
                             inference_func_args={'model': model})

def average_averages(args, size, model):
    return common.average_averages(times=args.num_tests, test_model_func=test_model, 
                                test_model_args={'args': args, 'size': size, 
                                                 'model': model})

def build_model(model_name, cn, cn_sac, cn_sq):
    if model_name == cn:
        return CornerNet()
    elif model_name == cn_sac:
        return CornerNet_Saccade()
    elif model_name == cn_sq:
        return CornerNet_Squeeze()

if __name__ == '__main__':
    args = parse_args()
    model = build_model(args.model, cn=CORNER_NET, 
                        cn_sac=CORNER_NET_SACCADE, cn_sq=CORNER_NET_SQUEEZE)
    out = average_averages(args=args, size=IMAGE_SIZE, model=model)
    common.print_output(args=args, out_data=out, model_name=args.model)
