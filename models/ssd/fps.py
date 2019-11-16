import torch
import os.path as path
import sys
from torch.autograd import Variable

sys.path.append(path.abspath('./ssd.pytorch'))
import ssd as model

sys.path.append('..')
import common

NET_NAME = 'SSD'
PHASE = 'test'
IMAGE_SIZE = 300
TRAINED_MODEL_DIR = 'ssd.pytorch/weights/'
TRAINED_MODEL_FN = 'ssd300_mAP_77.43_v2.pth'
TRAINED_MODEL_PATH = path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_FN)

def parse_args():
    """Parses arguments by calling up to common and then adding
       SSD specific arguments.

    Args:
         None
    
    Returns: parsed arguments
    """
    parser = common.default_args(net_name=NET_NAME, 
                                 num_classes=21, image_size=IMAGE_SIZE)
    parser.add_argument('--trained-model', required=False, help='Path to trained state_dict file', 
                         default=TRAINED_MODEL_PATH)
    return parser.parse_args()

def build_model(args, phase, size):
    """Builds an SSD model.

    Args:
         model_name: name of model

    Returns: model
    """
    ssd_model = model.build_ssd(phase, size, args.num_classes)
    ssd_model.load_state_dict(torch.load(args.trained_model))
    ssd_model.eval()
    ssd_model = ssd_model.cuda()

    return ssd_model

def inference(model, image, batch_size):
    """Executes common.time_inference function with SSD arguments.

    Args:
         model: SSD model
         image: input image
         batch_size: size of batch
    
    Returns: tuple
    """
    image = Variable(image)
    image = image.cuda()
    return common.time_inference(inference_func=model, 
                              inference_func_args={'x': image}, 
                                 batch_size=batch_size)

def read_data(args, size, batch_size):
    """Executes common.read_images with SSD arguments.

    Args:
         image_name_file: file with names of images
         image_path: path to images
         size: size of images
         batch_size: size of batch
    
    Returns: np array of shape (num_images, size, size, channels)
    """
    batches = common.read_images_batch(image_name_file=args.image_name_file, 
                                       image_path=args.image_path, size=size, 
                                       batch_size=batch_size)
    return common.prepare_images(images=batches, size=size, batch=batch_size)

def set_default_tensor_type():
    """Sets tensor type so SSD model does not error out.

    Args:
         None
    
    Returns: None
    """
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("ERROR: cuda is not available. Test will exit.")

if __name__ == '__main__':
    set_default_tensor_type()
    args = parse_args()

    model = build_model(args=args, phase=PHASE, size=IMAGE_SIZE)
    images = read_data(args=args, size=IMAGE_SIZE, batch_size=args.batch_size)
    avgs = common.default_average_averages(num_tests=args.num_tests, model=model, 
                            batch_size=args.batch_size, images=images, 
                                           inference_func=inference)

    common.print_output(num_tests=args.num_tests, out_data=avgs, model_name=NET_NAME + str(IMAGE_SIZE))
 
