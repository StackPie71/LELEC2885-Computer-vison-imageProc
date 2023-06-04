import os
import random
import sys
from typing import Optional
import torch
import numpy as np
import cv2
from IPython.display import Image, display
import matplotlib.pyplot as plt

def img_path_to_np_flt(fpath: str):
    """returns a numpy float32 array from RGB image path (8-16 bits per component)
    shape: c, h, w
    """
    if not os.path.isfile(fpath):
        raise ValueError(f"File not found {fpath}")
    try:
        rgb_img = cv2.cvtColor(
            cv2.imread(fpath, flags=cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH),
            cv2.COLOR_BGR2RGB,
        ).transpose(2, 0, 1)
    except cv2.error as e:
        print(f"img_path_to_np_flp: error {e} with {fpath}")
        breakpoint()
    if rgb_img.dtype == np.ubyte:
        return rgb_img.astype(np.single) / 255
    elif rgb_img.dtype == np.ushort:
        return rgb_img.astype(np.single) / 65535
    else:
        raise TypeError(
            "img_path_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})"
        )


def img_fpath_to_pt_tensor(fpath: str, batch: bool = True, crop_to_multiple: Optional[int] = None):
    """Open an image file path and convert it to PyTorch tensor."""
    tensor = torch.tensor(img_path_to_np_flt(fpath))
    if batch:
        tensor = tensor.unsqueeze(0)
    if crop_to_multiple:
        height, width = tensor.shape[-2:]
        height = (height // crop_to_multiple) * crop_to_multiple
        width = (width // crop_to_multiple) * crop_to_multiple
        tensor = tensor[..., :height, :width]
    return tensor
    
def display_pt_img(tensor_img, zoom: bool = False):
    """Display a tensor image."""
    zoom = True if 'google.colab' in sys.modules else zoom  # forcing zoom=True to work with google colab
    tensor_img = tensor_img.cpu()
    if tensor_img.dim() == 4:
        disp_img = tensor_img.squeeze(0)
    else:
        disp_img = tensor_img
    disp_img = disp_img.permute(1,2,0)
    if zoom:
        fig = plt.figure()
        plt.imshow(disp_img)
        display(fig)
        plt.close()
    else:
        disp_img = cv2.cvtColor(disp_img.numpy()*255, cv2.COLOR_RGB2BGR)
        disp_img = cv2.imencode('.jpg', disp_img)[1]
        display(Image(data=disp_img))

def get_random_testimg_fpath(category: str = 'misc'):
    """Return the path to a random image in test_images/<category>."""
    testimg_dpath = os.path.join('test_images', category)
    assert os.path.isdir(testimg_dpath), f'Directory does not exist: {testimg_dpath}'
    return os.path.join(testimg_dpath, random.choice(os.listdir(testimg_dpath)))
