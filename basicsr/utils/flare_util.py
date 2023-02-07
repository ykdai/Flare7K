import argparse
import json
import os
from cv2 import merge
import cv2
import skimage
from skimage import morphology
import torch
import numpy as np


_EPS=1e-7

def get_args_from_json(json_file_path):
    args_dict={}    
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict
    
def save_args_to_json(args_dict,json_folder_path='config/',file_name='config1.json'):
    json_file_path=json_folder_path+file_name
    args_dict = json.dumps(args_dict, indent=2, separators=(',', ':'))
    with open(json_file_path,"w") as f:
        #json.dump(args_dict,f)
        f.write(args_dict)

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def adjust_gamma(image: torch.Tensor, gamma):
    #image is in shape of [B,C,H,W] and gamma is in shape [B]
    gamma=gamma.float().cuda()
    gamma_tensor=torch.ones_like(image)
    gamma_tensor=gamma.view(-1,1,1,1)*gamma_tensor
    image=torch.pow(image,gamma_tensor)
    out= torch.clamp(image, 0.0, 1.0)
    return out

def adjust_gamma_reverse(image: torch.Tensor, gamma):
    #gamma=torch.Tensor([gamma]).cuda()
    gamma=1/gamma.float().cuda()
    gamma_tensor=torch.ones_like(image)
    gamma_tensor=gamma.view(-1,1,1,1)*gamma_tensor
    image=torch.pow(image,gamma_tensor)
    out= torch.clamp(image, 0.0, 1.0)
    return out

def predict_flare_from_6_channel(input_tensor,gamma):
    #the input is a tensor in [B,C,H,W], the C here is 6

    deflare_img=input_tensor[:,:3,:,:]
    flare_img_predicted=input_tensor[:,3:,:,:]

    merge_img_predicted_linear=adjust_gamma(deflare_img,gamma)+adjust_gamma(flare_img_predicted,gamma)
    merge_img_predicted=adjust_gamma_reverse(torch.clamp(merge_img_predicted_linear, 1e-7, 1.0),gamma)
    return deflare_img,flare_img_predicted,merge_img_predicted


def predict_flare_from_3_channel(input_tensor,flare_mask,base_img,flare_img,merge_img,gamma):
    #the input is a tensor in [B,C,H,W], the C here is 3
    
    input_tensor_linear=adjust_gamma(input_tensor,gamma)
    merge_tensor_linear=adjust_gamma(merge_img,gamma)
    flare_img_predicted=adjust_gamma_reverse(torch.clamp(merge_tensor_linear-input_tensor_linear, 1e-7, 1.0),gamma)

    masked_deflare_img=input_tensor*(1-flare_mask)+base_img*flare_mask
    masked_flare_img_predicted=flare_img_predicted*(1-flare_mask)+flare_img*flare_mask
    
    return masked_deflare_img,masked_flare_img_predicted


def get_highlight_mask(image, threshold=0.99,luminance_mode=False):
    """Get the area close to the exposure
    Args:
        image: the image tensor in [B,C,H,W]. For inference, B is set as 1.
        threshold: the threshold of luminance/greyscale of exposure region
        luminance_mode: use luminance or greyscale 
    Return:
        Binary image in [B,H,W]
    """
    if luminance_mode:
        # 3 channels in RGB
        luminance = 0.2126*image[:,0,:,:] + 0.7152*image[:,1,:,:] + 0.0722*image[:,2,:,:]
        binary_mask = luminance > threshold
    else:
        binary_mask = image.mean(dim=1, keepdim=True) > threshold
    binary_mask = binary_mask.to(image.dtype)
    return binary_mask

def _create_disk_kernel(kernel_size):
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    kernel = np.float32(rr <= np.max(x)) + _EPS
    kernel = kernel / np.sum(kernel)
    return kernel

def refine_mask(mask, morph_size = 0.01):
  """Refines a mask by applying mophological operations.
  Args:
    mask: A float array of shape [H, W]
    morph_size: Size of the morphological kernel relative to the long side of
      the image.

  Returns:
    Refined mask of shape [H, W].
  """
  mask_size = max(np.shape(mask))
  kernel_radius = .5 * morph_size * mask_size
  kernel = morphology.disk(np.ceil(kernel_radius))
  opened = morphology.binary_opening(mask, kernel)
  return opened

def blend_light_source(input_scene, pred_scene,threshold=0.99,luminance_mode=False):
    binary_mask = (get_highlight_mask(input_scene,threshold=threshold,luminance_mode=luminance_mode) > 0.5).to("cpu", torch.bool)
    binary_mask = binary_mask.squeeze()  # (h, w)
    binary_mask = binary_mask.numpy()
    binary_mask = refine_mask(binary_mask)

    labeled = skimage.measure.label(binary_mask)
    properties = skimage.measure.regionprops(labeled)
    max_diameter = 0
    for p in properties:
        # The diameter of a circle with the same area as the region.
        max_diameter = max(max_diameter, p["equivalent_diameter"])

    mask = np.float32(binary_mask)
    kernel_size = round(1.5 * max_diameter) #default is 1.5
    if kernel_size > 0:
        kernel = _create_disk_kernel(kernel_size)
        mask = cv2.filter2D(mask, -1, kernel)
        mask = np.clip(mask*3.0, 0.0, 1.0)
        mask_rgb = np.stack([mask] * 3, axis=0)

        mask_rgb = torch.from_numpy(mask_rgb).to(input_scene.device, torch.float32)
        blend = input_scene * mask_rgb + pred_scene * (1 - mask_rgb)
    else:
        blend = pred_scene
    return blend
