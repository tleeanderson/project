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
"""Inference implementations do not support batch inference
   https://github.com/princeton-vl/CornerNet-Lite/issues/105"""
BATCH_SIZE = 1

def parse_args():
    """Parses arguments by calling up to common and then adding
       CornerNet_Lite specific arguments.

    Args:
         None
    
    Returns: parsed arguments
    """
    parser = common.default_args(net_name=NET_NAME,  num_classes=81, image_size=IMAGE_SIZE)
    parser.add_argument('--model', required=False, default=CORNER_NET, 
                        choices=[CORNER_NET, CORNER_NET_SACCADE, CORNER_NET_SQUEEZE])
    return parser.parse_args()

def inference(model, image, batch_size):
    """Executes common.time_inference function with CornerNet_Lite arguments.

    Args:
         model: CornerNet_Lite model
         image: input image
         batch_size: size of batch
    
    Returns: tuple
    """
    return common.time_inference(inference_func=model, 
                                 inference_func_args={'image': image}, 
                                 batch_size=batch_size)

def read_data(image_name_file, image_path, size, batch_size):
    """Executes common.read_images with CornerNet_Lite arguments.

    Args:
         image_name_file: file with names of images
         image_path: path to images
         size: size of images
         batch_size: size of batch
    
    Returns: np array of shape (num_images, size, size, channels)
    """
    return common.read_images(image_name_file=image_name_file,
                              image_path=image_path, size=size)

def test_model(size, model, batch_size, images):
    """Executes common.test_model with CornerNet_Lite arguments.

    Args:
         size: size of model
         model: some CornerNet_Lite model
         batch_size: size of batch
         images: input images
    
    Returns: map
    """
    return common.test_model(im_data=images, inference_func=inference, 
                             inference_func_args={'model': model, 
                                                  'batch_size': batch_size})

def average_averages(size, model, batch_size, images):
    """Executes common.average_averages with CornerNet_Lite arguments.

    Args:
         size: size of images
         model: some CornerNet_Lite model
         batch_size: size of batch
         images: some images
    
    Returns: map
    """
    return common.average_averages(times=args.num_tests, test_model_func=test_model, 
                                test_model_args={'size': size, 'model': model, 
                                                 'batch_size': batch_size, 
                                                 'images': images})

def build_model(model_name):
    """Builds a model.

    Args:
         model_name: name of model

    Returns: model
    """
    if model_name == CORNER_NET:
        return CornerNet()
    elif model_name == CORNER_NET_SACCADE:
        return CornerNet_Saccade()
    elif model_name == CORNER_NET_SQUEEZE:
        return CornerNet_Squeeze()

if __name__ == '__main__':
    args = parse_args()
    model = build_model(args.model)
    images = read_data(image_name_file=args.image_name_file, image_path=args.image_path, 
                       size=args.image_size, batch_size=BATCH_SIZE)
    out = average_averages(size=args.image_size, model=model,
                           batch_size=BATCH_SIZE, images=images)
    common.print_output(num_tests=args.num_tests, out_data=out, model_name=args.model)
