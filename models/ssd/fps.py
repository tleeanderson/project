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
    parser = common.default_args(net_name=NET_NAME, 
                                 num_classes=21, image_size=IMAGE_SIZE)
    parser.add_argument('--trained-model', required=False, help='Path to trained state_dict file', 
                         default=TRAINED_MODEL_PATH)
    return parser.parse_args()

def build_model(args, phase, size):
    ssd_model = model.build_ssd(phase, size, args.num_classes)
    ssd_model.load_state_dict(torch.load(args.trained_model))
    ssd_model.eval()
    ssd_model = ssd_model.cuda()

    return ssd_model

def inference(model, image, batch_size):
    image = Variable(image)
    image = image.cuda()
    return common.time_inference(inference_func=model, 
                              inference_func_args={'x': image}, 
                                 batch_size=batch_size)

def read_data(args, size, batch_size):
    batches = common.read_images_batch(image_name_file=args.image_name_file, 
                                       image_path=args.image_path, size=size, 
                                       batch_size=batch_size)
    return common.prepare_images(images=batches, size=size, batch=batch_size)

def test_model(images, model, batch_size):
    return common.test_model(im_data=images, inference_func=inference, 
                          inference_func_args={'model': model, 
                                               'batch_size': batch_size})

def average_averages(times, model, batch_size, images):
    return common.average_averages(times=times, test_model_func=test_model, 
                                   test_model_args={'model': model, 
                                                    'batch_size': batch_size, 
                                                    'images': images})

def set_default_tensor_type():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("ERROR: cuda is not available. Test will exit.")

if __name__ == '__main__':
    set_default_tensor_type()
    args = parse_args()

    model = build_model(args=args, phase=PHASE, size=IMAGE_SIZE)
    images = read_data(args=args, size=IMAGE_SIZE, batch_size=args.batch_size)
    avgs = average_averages(times=args.num_tests, model=model, 
                            batch_size=args.batch_size, 
                            images=images)

    common.print_output(num_tests=args.num_tests, out_data=avgs, model_name=NET_NAME + str(IMAGE_SIZE))
 
