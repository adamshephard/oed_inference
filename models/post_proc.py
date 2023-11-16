import cv2
import numpy as np
import math

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
    binary_closing,
    binary_opening
)

from skimage.segmentation import watershed
from skimage import morphology
from misc.utils import get_bounding_box, remove_small_objects

import warnings


def noop(*args, **kargs):
    pass


warnings.warn = noop


####
def __proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=11) # ksize was 21 (for 40X images); I've changed to 11 for 20X images 
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=11) # ksize was 21 (for 40X images); I've changed to 11 for 20X images 

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F 
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


####
def process_layers(pred_layers, fx = 0.125):
    # script is specific to epithelium segmentation
    pred_layers = pred_layers.astype('uint32')
    w = pred_layers.shape[1]
    h = pred_layers.shape[0]

    # Zero-pad the array to ensure the morphological operations doesn't create an empty border
    pred_layers = np.pad(pred_layers, ((5, 5), (5, 5)), 'constant')

    def remove_holes_objects(img, size):
        img = morphology.remove_small_objects(img == 1, min_size=size, connectivity=2)
        img = morphology.remove_small_holes(img == True, area_threshold=size)
        return img

    def remove_and_dilate(img):
        img = remove_holes_objects(img)
        img = binary_dilation(img, morphology.disk(5))  # potentially change this
        return img

    def closing_opening(img, disk=5, size=750, fx = 0.124):
        disk = math.ceil(disk*(fx**2))
        size = int(size*(fx**2))

        img = binary_closing(img, morphology.disk(disk))
        img = binary_opening(img, morphology.disk(disk))
        img = remove_holes_objects(img, size)
        return img

    def crop_center(img, cropx, cropy):
        y,x = img.shape
        startx = x//2 - (cropx//2)
        starty = y//2 - (cropy//2)
        return img[starty:starty+cropy, startx:startx+cropx]

    t = np.where(pred_layers > 0, 1, 0)
    b = np.where(pred_layers == 2, 1, 0)
    e = np.where(pred_layers == 3, 1, 0)
    k = np.where(pred_layers == 4, 1, 0)

    t = closing_opening(t)
    b = closing_opening(b, size=750, fx=fx)
    e = closing_opening(e, size=2000, fx=fx)
    k = closing_opening(k, size=2000, fx=fx)

    layers = t + 2*b
    layers[layers == 3] = 2
    layers += 3*e
    layers[layers == 5] = 2  # basal takes priority of epithelium tissue
    layers[layers == 4] = 3  # epithelium takes priority of other tissue
    layers += 4*k
    layers[layers == 5] = 4  # keratin takes priority of other tissue
    layers[layers == 6] = 4  # allow keratin to take prioirty of basal - this may need to be changed!
    layers[layers == 7] = 4  # keratin takes priority of epithelium
    layers_cropped = crop_center(layers, w, h)
    return layers_cropped


def process_layers_new(ls_map: np.ndarray, fx = 0.125):
    ls_map = np.squeeze(ls_map)
    ls_map = np.around(ls_map).astype("uint8")  # ensure all numbers are integers
    min_size = 20000
    kernel_size = 10

    epith_all = np.where(ls_map >= 2, 1, 0).astype("uint8")
    mask = np.where(ls_map >= 1, 1, 0).astype("uint8")
    epith_all = epith_all > 0
    epith_mask = morphology.remove_small_objects(
        epith_all, min_size=min_size
    ).astype("uint8")
    epith_edited = epith_mask * ls_map
    epith_edited = epith_edited.astype("uint8")
    epith_edited_open = np.zeros_like(epith_edited).astype("uint8")
    for i in [3, 2, 4]:
        tmp = np.where(epith_edited == i, 1, 0).astype("uint8")
        ep_open = cv2.morphologyEx(
            tmp, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
        )
        ep_open = cv2.morphologyEx(
            ep_open, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size))
        )
        epith_edited_open[ep_open == 1] = i
    
    mask_open = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
    )
    mask_open = cv2.morphologyEx(
        mask_open, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size))
    ).astype("uint8")
    ls_map = mask_open.copy()
    for i in range(2, 5):
        ls_map[epith_edited_open == i] = i
    
    return ls_map.astype("uint8")

def process_epith(ls_map: np.ndarray, fx = 0.5):
    ls_map = np.squeeze(ls_map)
    ls_map = np.around(ls_map).astype("uint8")  # ensure all numbers are integers
    min_size = int(20000 * fx * fx)
    min_hole = int(1000 * fx * fx)
    kernel_size = int(11*fx)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # epith_labelled = morphology.label(ls_map, connectivity=2)
    epith_mask = morphology.remove_small_objects(
        ls_map.astype('bool'), min_size=min_size
    ).astype("uint8")

    ep_open = cv2.morphologyEx(
        epith_mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
    ).astype("uint8")
    ep_open = cv2.morphologyEx(
        ep_open, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size))
    ).astype(bool)

    ep_open = morphology.remove_small_holes(
        ep_open, min_hole).astype('uint8')

    return ep_open.astype("uint8")

####
def process(pred_map, nr_types=None, nr_layers=None, return_centroids=False):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        nr_layers: number of layers considered at output of ls branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_layer:    pixel-wise layer segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 

    """

    if nr_types is not None and nr_layers is not None:
        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:4]
        pred_type = pred_type.astype(np.int32)
        # pred_layer = process_layers(pred_map[..., 4])
        pred_layer = process_epith(pred_map[..., 4])
        # pred_layer = pred_map[..., 4]
        pred_layer = pred_layer.astype(np.int32)
    if nr_types is None and nr_layers is not None:
        pred_layer = pred_map
        # pred_layer = process_layers(pred_map)
        # pred_layer = process_epith(pred_map)
        pred_layer = pred_layer#.astype(np.int32)
        pred_inst = None
    if nr_types is not None and nr_layers is None:
        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:4]
        pred_type = pred_type.astype(np.int32)
        pred_layer = None
    if nr_types is None and nr_layers is None:
        pred_inst = pred_map
        pred_layer = None

    if nr_types is not None or nr_layers is None:
        pred_inst = np.squeeze(pred_inst)
        pred_inst = __proc_np_hv(pred_inst)

    inst_info_dict = None
    if nr_types is not None or nr_layers is None:
    # if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            # if nr_layers is not None:
            #     x, y = inst_info_dict[inst_id]["centroid"]
            #     if inst_type in [0, 2]:
            #         layer_type = pred_layer[int(y), int(x)]
            #         if layer_type == 1:
            #             inst_type = layer_type
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)
    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    return pred_inst, pred_layer, inst_info_dict
