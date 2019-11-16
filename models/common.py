import os.path as path
import cv2
import torch
import argparse
import json
import time
import numpy as np

IMAGE_NAME_FILE = '../top_600.txt'
KEY_IGNORE_SET = {'points'}

def default_args(net_name, num_classes, image_size, 
                 num_tests=10, image_name_file=IMAGE_NAME_FILE):
    """Given default values, creates default arguments via argparse.

    Args:
         net_name: name of network
         num_classes: number of classes
         image_size: size of image
         num_tests: number of times to run inference test
         image_name_file: file containing names of images
    
    Returns: parser
    """
    parser = argparse.ArgumentParser(description="Measure FPS of {}".format(net_name))
    parser.add_argument('--num-classes', required=False, type=int, help='number of classes', 
                        default=num_classes)
    parser.add_argument('--image-path', required=True, help='path to COCO val2014 images dir')
    parser.add_argument('--image-name-file', required=False, help='path to image name file', 
                        default=image_name_file)
    parser.add_argument('--image-size', required=False, type=int, default=image_size)
    parser.add_argument('--num-tests', required=False, type=int, default=num_tests)
    parser.add_argument('--batch-size', type=int, default=1)
    return parser

def read_file(path):
    """Given a path, reads in a file.

    Args:
         path: path to file
    
    Returns: data from disk
    """
    with open(path) as f:
        data = f.read()
    return data

def process_file_names(data):
    """Given string, splits on newline and returns
       everything but last element.

    Args:
         data: some string
    
    Returns: list
    """
    return data.split('\n')[:-1]

def resize_images(size, images):
    """Resizes images via cv2 resize.

    Args:
         size: desired size of image
         images: list of images
    
    Returns: list
    """
    return [cv2.resize(i, (size, size)) for i in images]

def prepare_images(images, size, batch):
    """Creates torch tensors from input images.

    Args:
         images: input images
         size: size of images
         batch: number of images in batch
    
    Returns: list
    """
    return [torch.from_numpy(i).unsqueeze(0)\
            .reshape((batch, 3, size, size)).float() for i in images]

def images_from_disk(im_path, names):
    """Reads images from disk.

    Args:
         im_path: path to images
         names: names of images at path
    
    Returns: tuple
    """
    images, nf = [], []
    for n in names:
        p = path.join(im_path, n)
        im = cv2.imread(p)
        if im is None:
            nf.append(p)
        else:
            images.append(im)
    return images, nf

def time_inference(inference_func, inference_func_args, batch_size):
    """Calls an inference func with its args and times
       the call. Returns both the time of the call and 
       the output.

    Args:
         inference_func: some inference function
         inference_func_args: some arguments to inference_func
         batch_size: size of batch
    
    Returns: tuple
    """
    start = time.time()
    out = inference_func(**inference_func_args)
    end = time.time()

    return (end - start) / batch_size, out

def average_averages(times, test_model_func, test_model_args, ik=KEY_IGNORE_SET):
    """Runs a given test_model_func n times and averages the
       results.

    Args:
         times: number of times to execute test_model_func
         test_model_func: some function to test a model
         test_model_args: some args to test_model_func
         ik: ignore keys, defaults to KEY_IGNORE_SET
         
    Returns: map
    """
    totals = {}
    for t in range(1, times + 1):
        print("On test number %d" % (t))
        out_m = test_model_func(**test_model_args)
        totals = {k: totals[k] + out_m[k] if k in totals else out_m[k]\
                  for k in set(out_m).difference(set(ik))}

    return {k: totals[k] / times for k in totals}

def batch_images(images, batch_size):
    """Batches images according to some batch_size.

    Args:
         images: input images
         batch_size: some batch size
    
    Returns: list of tensors
    """
    remainder = images.shape[0] % batch_size
    batches = range(0, images.shape[0] + 1, batch_size)
    result = []
    for bs, be in zip(range(len(batches)), range(1, len(batches))):
        result.append(images[batches[bs]:batches[be]])
    return result, images[images.shape[0]-remainder:]

def read_images_batch(image_name_file, image_path, size, batch_size):
    """Reads images from disk and batches them.

    Args:
         image_name_file: file with names of files
         image_path: path to images
         size: size of image
         batch_size: size of batch
    
    Returns: list of tensors
    """
    images = read_images(image_name_file=image_name_file, 
                                    image_path=image_path, size=size)
    batches, remainder = batch_images(images=images,
                                             batch_size=batch_size)
    print("With {} images and batch size of {}, remainder is {}. These will be left out of computation."\
          .format(len(images), batch_size, len(remainder)))
    return batches

def test_model(im_data, inference_func, inference_func_args):
    """Tests a given inference_func by invoking for each
       element in im_data. Each result contributes to
       an average FPS calculation.

    Args:
         im_data: image data
         inference_func: some inference function
         inference_func_args: arguments to inference_func
    
    Returns: map
    """
    points = []
    for i, img in enumerate(im_data):
        points.append(inference_func(image=img, **inference_func_args)[0])
    avg_sec = np.average(points)
    return {'avg_per_image_ms': avg_sec * 1000, 'avg_per_image_s': avg_sec, 
            'avg_fps': 1 / avg_sec, 'points': points}

def read_images(image_name_file, image_path, size):
    """Read images from disk.

    Args:
         image_name_file: file with image names
         image_path: path to images
         size: size of images
    
    Returns: np array of shape (num_images, size, size, channels)
    """
    images, nf = images_from_disk(im_path=image_path, 
                             names=process_file_names(read_file(image_name_file)))
    for p in nf:
        print("ERROR: Could not find {}".format(p))
    resized_images = resize_images(size=size, images=images)
    return np.asarray(resized_images)

def print_output(num_tests, out_data, model_name):
    """Prints given output.

    Args:
         num_tests: number of tests run
         out_data: output data
         model_name: name of model
    
    Returns: 
    """
    print('out_data ' + str(out_data))
    print("\n\nFrames Per Second result for %s, averaged over %d runs:" 
          % (str(model_name), num_tests))
    for k, v in out_data.items():
        print("%s = %f" % (str(k).ljust(25), v))
    print("\n\n")

def read_json_file(path):
    """Reads in a json file.

    Args:
         path: path to json file
    
    Returns: contents of json file
    """
    with open(path) as f:
        data = json.load(f)
    
    return data

def make_dirs(dirs):
    """Makes directories.

    Args:
         dirs: directories to make
    
    Returns: None
    """
    list(map(lambda d: os.makedirs(d, exist_ok=True), 
             list(filter(lambda d: not path.exists(d), dirs))))
