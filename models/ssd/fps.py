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

def inference(model, image):
    image = Variable(image)
    image = image.cuda()
    return common.time_inference(inference_func=model, 
                              inference_func_args={'x': image})

def test_model(args, size, model):
    images = common.prepare_images(images=common.read_images(image_name_file=args.image_name_file, 
                                                              image_path=args.image_path, 
                                                              size=size), size=size)
    return common.test_model(im_data=images, inference_func=inference, 
                          inference_func_args={'model': model})

def average_averages(args, phase, size, times, model):
    return common.average_averages(times=times, test_model_func=test_model, 
                                   test_model_args={'args': args, 'size': size, 
                                                    'model': model})

def set_default_tensor_type():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("ERROR: cuda is not available. Test will exit.")

if __name__ == '__main__':
    set_default_tensor_type()
    args = parse_args()

    model = build_model(args=args, phase=PHASE, size=IMAGE_SIZE)
    avgs = average_averages(args=args, phase=PHASE, size=IMAGE_SIZE, 
                            times=args.num_tests, model=model)

    common.print_output(args=args, out_data=avgs, model_name=NET_NAME + str(IMAGE_SIZE))
 
