import sys
import torch
import time
import os.path as path

sys.path.append(path.abspath('./Retina_Net'))

sys.path.append('..')
import common

if __name__ == '__main__':
    retina_net = torch.load('../../pretrained/coco_resnet_50_map_0_335.pt')
    img = cv2.imread('/home/tanderson/models/datasets/mscoco/raw-data/val2014/COCO_val2014_000000000042.jpg')
    start = time.time()
    img = common.prepare_images(images=[img], size=512)[0].cuda()
    out = retina_net()

    print("time taken: {}".format(end - start))
