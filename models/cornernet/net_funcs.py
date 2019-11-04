import torch
import numpy as np
import os
import cv2
import sys

sys.path.append('./CornerNet')
from external.nms import soft_nms, soft_nms_merge
from utils import crop_image, normalize_

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def kp_decode(nnet, images, K, ae_threshold=0.5, kernel=3):
    detections = nnet.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel)
    detections = detections.data.cpu().numpy()
    return detections

def inference(dataset, nnet, image, decode_func=kp_decode):
    K = dataset.configs["top_k"]
    ae_threshold = dataset.configs["ae_threshold"]
    nms_kernel = dataset.configs["nms_kernel"]
    scales = dataset.configs["test_scales"]
    weight_exp = dataset.configs["weight_exp"]
    merge_bbox = dataset.configs["merge_bbox"]
    categories = dataset.configs["categories"]
    nms_threshold = dataset.configs["nms_threshold"]
    max_per_image = dataset.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[dataset.configs["nms_algorithm"]]

    top_bboxes = {}
    height, width = image.shape[0:2]
    detections = []
    for scale in scales:
        new_height = int(height * scale)
        new_width  = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width  = new_width  | 127

        images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios  = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes   = np.zeros((1, 2), dtype=np.float32)

        out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
        height_ratio = out_height / inp_height
        width_ratio  = out_width  / inp_width

        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, 
                                                   [inp_height, inp_width])

        resized_image = resized_image / 255.
        normalize_(resized_image, dataset.mean, dataset.std)

        images[0]  = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0]   = [int(height * scale), int(width * scale)]
        ratios[0]  = [height_ratio, width_ratio]

        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        dets   = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
        dets   = dets.reshape(2, -1, 8)
        dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
        dets   = dets.reshape(1, -1, 8)

        _rescale_dets(dets, ratios, borders, sizes)
        dets[:, :, 0:4] /= scale
        detections.append(dets)

    detections = np.concatenate(detections, axis=1)

    classes    = detections[..., -1]
    classes    = classes[0]
    detections = detections[0]

    # reject detections with negative scores
    keep_inds  = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes    = classes[keep_inds]

    top_bboxes = {}
    for j in range(categories):
        keep_inds = (classes == j)
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        if merge_bbox:
            soft_nms_merge(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm,
                           weight_exp=weight_exp)
        else:
            soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm)
        top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]

    scores = np.hstack([top_bboxes[j][:, -1] for j in range(1, categories + 1)])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds     = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]
    return top_bboxes
