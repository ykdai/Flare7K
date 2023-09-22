import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops
from basicsr.data.flare7k_dataset import Flare_Image_Loader,RandomGammaCorrection
from basicsr.archs.uformer_arch import Uformer
import argparse
from basicsr.archs.unet_arch import U_Net
from basicsr.utils.flare_util import blend_light_source,get_args_from_json,save_args_to_json,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
from torch.distributions import Normal
import torchvision.transforms as transforms
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,default=None)
parser.add_argument('--output',type=str,default=None)
parser.add_argument('--model_type',type=str,default='Uformer')
parser.add_argument('--model_path',type=str,default='checkpoint/flare7kpp/net_g_last.pth')
parser.add_argument('--output_ch',type=int,default=6)
parser.add_argument('--flare7kpp', action='store_const', const=True, default=False) #use flare7kpp's inference method and output the light source directly.

args = parser.parse_args()
model_type=args.model_type
images_path=os.path.join(args.input,"*.*")
result_path=args.output
pretrain_dir=args.model_path
output_ch=args.output_ch

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def load_params(model_path):
     full_model=torch.load(model_path)
     if 'params_ema' in full_model:
          return full_model['params_ema']
     elif 'params' in full_model:
          return full_model['params']
     else:
          return full_model

class ImageProcessor:
    def __init__(self, model):
        self.model = model

    def resize_image(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width < original_height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height))

    def process_image(self, image):
        # Open the original image
        to_tensor=transforms.ToTensor()
        original_image = image

        # Resize the image proportionally to make the shorter side 512 pixels
        resized_image = self.resize_image(original_image, 512)

        # Get the resized image's size
        resized_width, resized_height = resized_image.size

        cropped_image1 = resized_image.crop((0, 0, 512, 512))
        cropped_image2 = resized_image.crop((resized_width - 512, resized_height - 512, resized_width, resized_height))

        # Convert PIL images to NumPy arrays
        image_array1 = np.array(cropped_image1)
        image_array2 = np.array(cropped_image2)

        # Process the two cropped images using the model
        processed_image1 = self.model(to_tensor(image_array1).unsqueeze(0).cuda()).squeeze(0)
        processed_image2 = self.model(to_tensor(image_array2).unsqueeze(0).cuda()).squeeze(0)
        # Apply interpolation to the overlapped region
        if resized_width > 512:
            overlap_width = 512 - (resized_width - 512)
            alpha = torch.linspace(0, 1, steps=overlap_width).view(1, overlap_width, 1).expand(512, overlap_width, 6).permute(2,0,1).cuda()
            merged_image = alpha * processed_image2[:,:, :overlap_width] + (1 - alpha) * processed_image1[:,:, -overlap_width:]
        else:
            overlap_height = 512 - (resized_height - 512)
            alpha = torch.linspace(0, 1, steps=overlap_height).view(overlap_height, 1, 1).expand(overlap_height, 512, 6).permute(2,0,1).cuda()
            merged_image = alpha * processed_image2[:,:overlap_height] + (1 - alpha) * processed_image1[:,-overlap_height:]

        # Concatenate the non-overlapping regions
        if resized_width > 512:
            merged_image = torch.cat((processed_image1[:,:, :512-overlap_width], merged_image, processed_image2[:,:, overlap_width:]), dim=2)
        else:
            merged_image = torch.cat((processed_image1[:,:512-overlap_height], merged_image, processed_image2[:,overlap_height:]), dim=1)

        return merged_image

def demo(images_path,output_path,model_type,output_ch,pretrain_dir,flare7kpp_flag):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path=glob.glob(images_path)
    result_path=output_path
    torch.cuda.empty_cache()
    if model_type=='Uformer':
        model=Uformer(img_size=512,img_ch=3,output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type=='U_Net' or model_type=='U-Net':
        model=U_Net(img_ch=3,output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    else:
        assert False, "This model is not supported!!"
    processor=ImageProcessor(model)
    to_tensor=transforms.ToTensor()

    for i,image_path in tqdm(enumerate(test_path)):
        if not flare7kpp_flag:
            mkdir(os.path.join(result_path,"deflare/"))
            deflare_path = os.path.join(result_path,"deflare/",str(i).zfill(5)+"_deflare.jpg")

        mkdir(os.path.join(result_path,"flare/"))
        mkdir(os.path.join(result_path,"blend/"))
        
        #PIL will be faster while saving with JPG rather than PNG for large images
        flare_path = os.path.join(result_path,"flare/",str(i).zfill(5)+"_flare.jpg")
        blend_path = os.path.join(result_path,"blend/",str(i).zfill(5)+"_blend.jpg")

        merge_img = Image.open(image_path).convert("RGB")

        model.eval()
        with torch.no_grad():
            output_img=processor.process_image(merge_img).unsqueeze(0)
            gamma=torch.Tensor([2.2])
            if output_ch==6:
                deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(output_img,gamma)
            elif output_ch==3:
                flare_mask=torch.zeros_like(merge_img)
                deflare_img,flare_img_predicted=predict_flare_from_3_channel(output_img,flare_mask,output_img,merge_img,merge_img,gamma)
            else:
                assert False, "This output_ch is not supported!!"
            
            if not flare7kpp_flag:
                torchvision.utils.save_image(deflare_img, deflare_path)
                deflare_img= blend_light_source(to_tensor(processor.resize_image(merge_img,512)).cuda().unsqueeze(0), deflare_img, 0.95)                
            deflare_img_np=deflare_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            deflare_img_pil=Image.fromarray((deflare_img_np * 255).astype(np.uint8))
            flare_img_orig=ImageChops.difference(merge_img.resize(deflare_img_pil.size),deflare_img_pil)
            deflare_img_orig=ImageChops.difference(merge_img,flare_img_orig.resize(merge_img.size,resample=Image.BICUBIC))
            flare_img_orig.save(flare_path)
            deflare_img_orig.save(blend_path)
            
demo(images_path,result_path,model_type,output_ch,pretrain_dir,args.flare7kpp)