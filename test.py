import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
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
    to_tensor=transforms.ToTensor()
    resize=transforms.Resize((512,512)) #The output should in the shape of 128X
    for i,image_path in tqdm(enumerate(test_path)):
        if not flare7kpp_flag:
            mkdir(result_path+"deflare/")
            deflare_path = result_path+"deflare/"+str(i).zfill(5)+"_deflare.png"

        mkdir(result_path+"flare/")
        mkdir(result_path+"input/")
        mkdir(result_path+"blend/")
        
        flare_path = result_path+"flare/"+str(i).zfill(5)+"_flare.png"
        merge_path = result_path+"input/"+str(i).zfill(5)+"_input.png"
        blend_path = result_path+"blend/"+str(i).zfill(5)+"_blend.png"

        merge_img = Image.open(image_path).convert("RGB")
        merge_img = resize(to_tensor(merge_img))
        merge_img = merge_img.cuda().unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output_img=model(merge_img)
            #if ch is 6, first three channels are deflare image, others are flare image
            #if ch is 3, unsaturated region of output is the deflare image.
            gamma=torch.Tensor([2.2])
            if output_ch==6:
                deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(output_img,gamma)
            elif output_ch==3:
                flare_mask=torch.zeros_like(merge_img)
                deflare_img,flare_img_predicted=predict_flare_from_3_channel(output_img,flare_mask,output_img,merge_img,merge_img,gamma)
            else:
                assert False, "This output_ch is not supported!!"

            torchvision.utils.save_image(merge_img, merge_path)
            torchvision.utils.save_image(flare_img_predicted, flare_path)
            if flare7kpp_flag:
                 torchvision.utils.save_image(deflare_img, blend_path)
            else:
                blend_img= blend_light_source(merge_img, deflare_img, 0.97)
                torchvision.utils.save_image(deflare_img, deflare_path)
                torchvision.utils.save_image(blend_img, blend_path)
       
demo(images_path,result_path,model_type,output_ch,pretrain_dir,args.flare7kpp)