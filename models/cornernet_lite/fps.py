sys.path.append('..')
import common

def parse_args():
    parser = common.default_args(net_name=NET_NAME, trained_model_path=TRAINED_MODEL_PATH, 
                              num_classes=81, image_size=IMAGE_SIZE)
    parser.add_argument('--model-config', required=False, default='CornerNet')
    parser.add_argument('--suffix', required=False, default='.json')
    return parser.parse_args()

if __name__ == '__main__':
    
