import sys
import os.path as path

sys.path.append(path.abspath('./keras_retina_net'))
import keras_retinanet.models as models

sys.path.append('..')
import common

MODEL_PATH = '../../pretrained/resnet50_coco_best_v2.1.0.h5'
NET_NAME = 'RetinaNet'
IMAGE_SIZE = 512

def parse_args():
    """Parses arguments by calling up to common and then adding
       RetinaNet specific arguments.

    Args:
         None
    
    Returns: parsed arguments
    """
    parser = common.default_args(net_name=NET_NAME, num_classes=81, image_size=IMAGE_SIZE)
    parser.add_argument('--model-path', default=MODEL_PATH)

    return parser.parse_args()

def build_model(path):
    """Builds a retinanet model given a path.

    Args:
         path: path to pretrained retinanet model
    
    Returns: model
    """
    return models.load_model(filepath=path)

def inference(model, image, batch_size):
    """Executes common.time_inference function with RetinaNet arguments.

    Args:
         model: RetinaNet model
         image: input image
         batch_size: size of batch
    
    Returns: tuple
    """
    return common.time_inference(inference_func=model.predict_on_batch, 
                                 inference_func_args={'x': image}, 
                                 batch_size=batch_size)

def read_data(image_name_file, image_path, size, batch_size):
    """Executes common.read_images with RetinaNet arguments.

    Args:
         image_name_file: file with names of images
         image_path: path to images
         size: size of images
         batch_size: size of batch
    
    Returns: np array of shape (num_images, size, size, channels)
    """
    return common.read_images_batch(image_name_file=image_name_file, 
                                    image_path=image_path, 
                                    size=size, batch_size=batch_size)

def test_model(images, model, batch_size):
    """Executes common.test_model with RetinaNet arguments.

    Args:
         size: size of model
         model: some RetinaNet model
         batch_size: size of batch
         images: input images
    
    Returns: map
    """
    return common.test_model(im_data=images, inference_func=inference, 
                          inference_func_args={'model': model,
                                               'batch_size': batch_size})

def average_averages(model, batch_size, images):
    """Executes common.average_averages with RetinaNet arguments.

    Args:
         size: size of images
         model: some RetinaNet model
         batch_size: size of batch
         images: some images
    
    Returns: map
    """
    return common.average_averages(times=args.num_tests, test_model_func=test_model, 
                                   test_model_args={'model': model, 
                                                    'images': images,
                                                    'batch_size': batch_size})

if __name__ == '__main__':
    args = parse_args()
    model = build_model(path=MODEL_PATH)
    model.summary()
    images = read_data(image_name_file=args.image_name_file, 
                       image_path=args.image_path, size=args.image_size, 
                       batch_size=args.batch_size)
    avgs = average_averages(model=model, batch_size=args.batch_size, 
                            images=images)
    common.print_output(num_tests=args.num_tests, out_data=avgs, 
                        model_name=NET_NAME + str(IMAGE_SIZE))
    
