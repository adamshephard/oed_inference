import math
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from scipy import ndimage
from scipy.ndimage import measurements
from skimage import morphology as morph
import matplotlib.pyplot as plt

from misc.utils import center_pad_to_shape, cropping_center, get_bounding_box
from dataloader.augs import fix_mirror_padding


def gen_instance_cnt_map(ann, ksize, crop_shape):
    """Input annotation must be of original shape"""
    orig_ann = ann.copy()  # instance ID map
    fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map
    crop_ann = cropping_center(fixed_ann, crop_shape)

    # setting 1 boundary pix of each instance to background
    inner_map = np.zeros(fixed_ann.shape[:2], np.uint8)
    contour_map = np.zeros(fixed_ann.shape[:2], np.uint8)

    inst_list = list(np.unique(crop_ann))
    if 0 in inst_list:
        inst_list.remove(0)  # 0 is background

    # get structuring element
    k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))

    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inner = cv2.erode(inst_map, k_disk, iterations=1)
        outer = cv2.dilate(inst_map, k_disk, iterations=1)
        inner_map += inner
        contour_map += outer - inner

    inner_map[inner_map > 0] = 1  # binarize
    contour_map[contour_map > 0] = 1  # binarize
    bg_map = 1 - (inner_map + contour_map)
    return np.dstack([bg_map, inner_map, contour_map])


####
def gen_instance_hv_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = ann.copy()  # instance ID map
    fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map
    crop_ann = cropping_center(fixed_ann, crop_shape)
    # TODO: deal with 1 label warning
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map


####
def gen_targets(ann, crop_shape, **kwargs):
    """Generate the targets for the network."""
    seg_mode = kwargs['seg_mode']
    model_name = kwargs['model_name']
    nr_types = kwargs['nr_types']
    contour_ksize = kwargs['contour_ksize']

    ann = ann.copy()

    # inst_map = ann[..., 0]
    # type_map = ann[..., 1]
    # ls_map = ann[..., 2]
    ls_map = ann[..., 0]
    ls_map[ls_map == 1] = 0 #2] = 0
    ls_map[ls_map == 2] = 1 #3] = 2

    target_dict = {}

    if seg_mode in ['instance', 'multi']:
        if 'hovernet' in model_name and nr_types != None:
            hv_map = gen_instance_hv_map(inst_map, crop_shape)
            np_map = inst_map.copy()
            np_map[np_map > 0] = 1
            if nr_types == 3:
                type_map[type_map == 3] = 2  # merge all epithelial layers
                type_map[type_map == 4] = 2  # merge all epithelial layers
            tp_map = cropping_center(type_map, crop_shape)
            hv_map = cropping_center(hv_map, crop_shape)
            np_map = cropping_center(np_map, crop_shape)
            target_dict["tp_map"] = tp_map
            target_dict["hv_map"] = hv_map
            target_dict["np_map"] = np_map
        
        elif model_name == 'unet':
        #     np_map = inst_map.copy()
        #     np_map[np_map > 0] = 1
        #     tp_map = cropping_center(type_map, crop_shape)
        #     np_map = cropping_center(np_map, crop_shape)
        #     target_dict["tp_map"] = tp_map
        #     target_dict["np_map"] = np_map            

        # else:
            # np_map = inst_map.copy()
            # np_map[np_map > 0] = 1
            inst = gen_instance_cnt_map(inst_map, contour_ksize, crop_shape)
            inst = cropping_center(inst, crop_shape)
            inst = inst / 100 # fix scaling
            inst = inst.astype('int')
            target_dict["inst"] = inst
            # target_dict["tp_map"] = tp_map
    
    if seg_mode in ['semantic', 'multi']:
        ls_map = cropping_center(ls_map, crop_shape)
        target_dict["ls_map"] = ls_map
    return target_dict


####
def prep_sample(data, **kwargs):
    shape_array = [np.array(v.shape[:2]) for v in data.values()]
    shape = np.maximum(*shape_array)

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        ch = np.squeeze(ch.astype("float32"))
        ch = ch / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        ch_cmap = center_pad_to_shape(ch_cmap, shape)
        return ch_cmap

    viz_list = []
    # cmap may randomly fails if of other types
    viz_list.append(colorize(data["np_map"], 0, 1))
    # map to [0,2] for better visualisation.
    # Note, [-1,1] is used for training.
    viz_list.append(colorize(data["hv_map"][..., 0] + 1, 0, 2))
    viz_list.append(colorize(data["hv_map"][..., 1] + 1, 0, 2))
    img = center_pad_to_shape(data["img"], shape)
    return np.concatenate([img] + viz_list, axis=1)
