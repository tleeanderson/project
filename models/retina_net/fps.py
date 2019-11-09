import sys
import os.path as path

sys.path.append(path.abspath('./keras_retina_net'))
import keras_retinanet.models as models

sys.path.append('..')
import common
import numpy as np

MODEL_PATH = '../../pretrained/resnet50_coco_best_v2.1.0.h5'
NET_NAME = 'RetinaNet'
IMAGE_SIZE = 512

def parse_args():
    parser = common.default_args(net_name=NET_NAME, num_classes=81, image_size=IMAGE_SIZE)
    parser.add_argument('--pretrained-model', default=MODEL_PATH)

    return parser.parse_args()

def build_model(path):
    return models.load_model(filepath=path)

def inference(model, image):
    return common.time_inference(inference_func=model.predict_on_batch, 
                                 inference_func_args={'x': np.expand_dims(image, axis=0)})

def test_model(args, size, model):
    images = common.read_images(image_name_file=args.image_name_file, 
                                                              image_path=args.image_path, 
                                                              size=size)
    return common.test_model(im_data=images, inference_func=inference, 
                          inference_func_args={'model': model})

def average_averages(args, size, model):
    return common.average_averages(times=args.num_tests, test_model_func=test_model, 
                                   test_model_args={'args': args, 'size': size, 
                                                    'model': model})

if __name__ == '__main__':
    args = parse_args()
    model = build_model(path=MODEL_PATH)
    avgs = average_averages(args=args, size=IMAGE_SIZE, model=model)
    common.print_output(args=args, out_data=avgs, model_name=NET_NAME + str(IMAGE_SIZE))
    
