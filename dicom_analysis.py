import pdb
from functools import partial
import pandas as pd
from PIL import Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import skimage
import os
import sys
import pydicom
from typing import cast, List

from data_utils import load_split_metadf
from constants import PROJECT_DIR, R_LABELS, CXP_JPG_DIR, MIMIC_JPG_DIR, MODEL_SAVE_DIR, CXP_LABELS


def extract_window_center_width(ds, multi_index=0):
    elem = ds['WindowCenter']
    center = (
        cast(List[float], elem.value)[multi_index] if elem.VM > 1 else elem.value
    )
    center = cast(float, center)

    elem = ds['WindowWidth']
    width = cast(List[float], elem.value)[multi_index] if elem.VM > 1 else elem.value
    width = cast(float, width)

    return center, width


def window_dcm_im(ds, width_mult=None):
    # adapted from pydicom
    center, width = extract_window_center_width(ds)
    if width_mult is not None:
        width *= width_mult

    bits_stored = cast(int, ds.BitsStored)

    y_min = 0
    y_max = 2 ** bits_stored - 1

    y_range = y_max - y_min
    x = ds.pixel_array.astype('float64')

    below = x <= (center - width / 2)
    above = x > (center + width / 2)
    between = np.logical_and(~below, ~above)

    x[below] = y_min
    x[above] = y_max
    if between.any():
        x[between] = (
                ((x[between] - center) / width + 0.5) * y_range + y_min
        )

    if ds.PhotometricInterpretation == 'MONOCHROME1':
        x = np.clip(x.max() - x, 0, None)

    return x

