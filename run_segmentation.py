import os
import glob
import shutil
import numpy as np
from scipy import ndimage
from skimage import morphology
import cv2

from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
    WSIStreamDataset,
)
from tiatoolbox.utils.misc import imwrite
from models.model import TransUNet

def wsi_post_proc(wsi_seg: np.ndarray, fx: int=1) -> np.ndarray:
    """
    Post processing for WSI-level segmentations.
    """
    wsi_seg = np.around(wsi_seg).astype("uint8")  # ensure all numbers are integers
    min_size = int(30000 * fx)
    min_hole = int(8000)
    kernel_size = int(5*fx)
    if kernel_size % 2 == 0:
        kernel_size += 1
    # epith_labelled = morphology.label(ls_map, connectivity=2)
    epith_mask = morphology.remove_small_objects(
        wsi_seg.astype('bool'), min_size=min_size
    ).astype("uint8")
    ep_open = cv2.morphologyEx(
        epith_mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
    ).astype("uint8")
    ep_open = cv2.morphologyEx(
        ep_open, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size))
    ).astype(bool)
    ep_open = morphology.remove_small_holes(
        ep_open, min_hole).astype('uint8')
    return ep_open

def process_segmentation(seg_path: str, out_path: str, colour_dict: dict) -> None:
    """
    Post-processing for WSI-level segmentations.
    """
    seg = np.load(seg_path)
    seg = wsi_post_proc(seg)
    seg_col = np.expand_dims(seg, axis=2)
    seg_col = np.repeat(seg_col, 3, axis=2)
    for key, value in colour_dict.items():
        seg_col[seg == value[0]] = value[1]
    seg_col = seg_col.astype('uint8')
    imwrite(out_path, seg_col)
    return None
    

ON_GPU = True
"""ON_GPU should be True if cuda-enabled GPU is
available and False otherwise. However, it currently crashes without GPU"""
WORKERS = 0
if ON_GPU:
    WORKERS = 10

# Define input/output folders. Inputs files can be WSIs or images.
input_dir = "/data/ANTICIPATE/dyplasia_detection/oed_segmentation/input/"
output_dir = "/data/ANTICIPATE/dyplasia_detection/oed_segmentation/output/"
weights_path = "/data/ANTICIPATE/dyplasia_detection/oed_segmentation/pretrained/transunet_external.tar"
mode = "wsi" # or roi
colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "dysplasia": [1, [255, 0,   0]]
    }

# Defining ioconfig
iostate = IOSegmentorConfig(
    input_resolutions=[
        {"units": "mpp", "resolution": 1.0},
    ],
    output_resolutions=[
        {"units": "mpp", "resolution": 1.0},
    ],
    patch_input_shape=[512, 512],
    patch_output_shape=[384, 384], # performs central cropping
    stride_shape=[384, 384],
    save_resolution={"units": "mpp", "resolution": 1.0},
)

# Process
file_list = glob.glob(input_dir + "*")

# Creating the model
model = TransUNet(
    weights=weights_path,
    nr_layers=2,
    img_size=512,
)

segmentor = SemanticSegmentor(
    model=model,
    num_loader_workers=WORKERS,
    batch_size=1,
)

# Prediction
output = segmentor.predict(
    imgs=file_list,
    save_dir=os.path.join(output_dir, "transunet/tmp"),
    mode=mode,
    on_gpu=ON_GPU,
    crash_on_exception=True,
    ioconfig=iostate,
)

# check model/ visualise?

# Rename TIAToolbox output files to readability
out_dir = os.path.join(output_dir, "transunet")
os.makedirs(out_dir, exist_ok=True)

for out in output:
    basename = os.path.basename(out[0]).split(".")[0]
    outname = os.path.basename(out[1]).split(".")[0]
    process_segmentation(
        seg_path=os.path.join(output_dir, "transunet/tmp", f"{outname}.raw.0.npy"),
        out_path=os.path.join(out_dir, basename + ".png"),
        colour_dict=colour_dict,
        )
    shutil.rmtree(os.path.join(output_dir, "transunet/tmp"))