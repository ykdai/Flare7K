import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random

import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch
from basicsr.utils.registry import DATASET_REGISTRY

class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma
	def __call__(self,image):
		if self.gamma == None:
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)

def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image

def glod_from_folder(folder_list, index_list):
	ext = ['png','jpeg','jpg','bmp','tif']
	index_dict={}
	for i,folder_name in enumerate(folder_list):
		data_list=[]
		[data_list.extend(glob.glob(folder_name + '/*.' + e)) for e in ext]
		data_list.sort()
		index_dict[index_list[i]]=data_list
	return index_dict

class Flare_Image_Loader(data.Dataset):
	def __init__(self, image_path ,transform_base,transform_flare,mask_type=None):
		self.ext = ['png','jpeg','jpg','bmp','tif']
		self.data_list=[]
		[self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
		self.flare_dict={}
		self.flare_list=[]
		self.flare_name_list=[]

		self.reflective_flag=False
		self.reflective_dict={}
		self.reflective_list=[]
		self.reflective_name_list=[]

		self.mask_type=mask_type #It is a str which may be None,"luminance" or "color"
		self.img_size=transform_base['img_size']
		self.transform_base=transforms.Compose([transforms.RandomCrop((self.img_size,self.img_size),pad_if_needed=True,padding_mode='reflect'),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])

		self.transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(transform_flare['scale_min'],transform_flare['scale_max']),translate=(transform_flare['translate']/1440,transform_flare['translate']/1440),shear=(-transform_flare['shear'],transform_flare['shear'])),
									transforms.CenterCrop((self.img_size,self.img_size)),
									transforms.RandomHorizontalFlip(),
									transforms.RandomVerticalFlip()
									])

		print("Base Image Loaded with examples:", len(self.data_list))

	def __getitem__(self, index):
		# load base image
		img_path=self.data_list[index]
		base_img= Image.open(img_path).convert('RGB')
		
		gamma=np.random.uniform(1.8,2.2)
		to_tensor=transforms.ToTensor()
		adjust_gamma=RandomGammaCorrection(gamma)
		adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
		color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
		if self.transform_base is not None:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
			base_img=self.transform_base(base_img)
		else:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
		sigma_chi=0.01*np.random.chisquare(df=1)
		base_img=Normal(base_img,sigma_chi).sample()
		gain=np.random.uniform(0.5,1.2)
		flare_DC_offset=np.random.uniform(-0.02,0.02)
		base_img=gain*base_img
		base_img=torch.clamp(base_img,min=0,max=1)

		#load flare image
		flare_path=random.choice(self.flare_list)
		flare_img =Image.open(flare_path).convert('RGB')
		if self.reflective_flag:
			reflective_path=random.choice(self.reflective_list)
			reflective_img =Image.open(reflective_path).convert('RGB')


		flare_img=to_tensor(flare_img)
		flare_img=adjust_gamma(flare_img)
		
		if self.reflective_flag:
			reflective_img=to_tensor(reflective_img)
			reflective_img=adjust_gamma(reflective_img)
			flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)

		flare_img=remove_background(flare_img)

		if self.transform_flare is not None:
			flare_img=self.transform_flare(flare_img)
		
		#change color
		flare_img=color_jitter(flare_img)

		#flare blur
		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
		flare_img=blur_transform(flare_img)
		flare_img=flare_img+flare_DC_offset
		flare_img=torch.clamp(flare_img,min=0,max=1)

		#merge image	
		merge_img=flare_img+base_img
		merge_img=torch.clamp(merge_img,min=0,max=1)

		if self.mask_type==None:
			return {'gt': adjust_gamma_reverse(base_img),'flare': adjust_gamma_reverse(flare_img),'lq': adjust_gamma_reverse(merge_img),'gamma':gamma}
		elif self.mask_type=="luminance":
			#calculate mask (the mask is 3 channel)
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			luminance=0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
			threshold_value=0.99**gamma
			flare_mask=torch.where(luminance >threshold_value, one, zero)

		elif self.mask_type=="color":
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			threshold_value=0.99**gamma
			flare_mask=torch.where(merge_img >threshold_value, one, zero)
		elif self.mask_type=="flare":
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			threshold_value=0.7**gamma
			flare_mask=torch.where(flare_img >threshold_value, one, zero)
		return {'gt': adjust_gamma_reverse(base_img),'flare': adjust_gamma_reverse(flare_img),'lq': adjust_gamma_reverse(merge_img),'mask': flare_mask,'gamma': gamma}

	def __len__(self):
		return len(self.data_list)
	
	def load_scattering_flare(self,flare_name,flare_path):
		flare_list=[]
		[flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
		self.flare_name_list.append(flare_name)
		self.flare_dict[flare_name]=flare_list
		self.flare_list.extend(flare_list)
		len_flare_list=len(self.flare_dict[flare_name])
		if len_flare_list == 0:
			print("ERROR: scattering flare images are not loaded properly")
		else:
			print("Scattering Flare Image:",flare_name, " is loaded successfully with examples", str(len_flare_list))
		print("Now we have",len(self.flare_list),'scattering flare images')

	def load_reflective_flare(self,reflective_name,reflective_path):
		self.reflective_flag=True
		reflective_list=[]
		[reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
		self.reflective_name_list.append(reflective_name)
		self.reflective_dict[reflective_name]=reflective_list
		self.reflective_list.extend(reflective_list)
		len_reflective_list=len(self.reflective_dict[reflective_name])
		if len_reflective_list == 0:
			print("ERROR: reflective flare images are not loaded properly")
		else:
			print("Reflective Flare Image:",reflective_name, " is loaded successfully with examples", str(len_reflective_list))
		print("Now we have",len(self.reflective_list),'refelctive flare images')

@DATASET_REGISTRY.register()
class Flare_Pair_Loader(Flare_Image_Loader):
	def __init__(self, opt):
		Flare_Image_Loader.__init__(self,opt['image_path'],opt['transform_base'],opt['transform_flare'],opt['mask_type'])
		scattering_dict=opt['scattering_dict']
		reflective_dict=opt['reflective_dict']
		if len(scattering_dict) !=0:
			for key in scattering_dict.keys():
				self.load_scattering_flare(key,scattering_dict[key])
		if len(reflective_dict) !=0:
			for key in reflective_dict.keys():
				self.load_reflective_flare(key,reflective_dict[key])

@DATASET_REGISTRY.register()
class Image_Pair_Loader(data.Dataset):
    def __init__(self, opt):
        super(Image_Pair_Loader, self).__init__()
        self.opt = opt
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = glod_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        self.to_tensor=transforms.ToTensor()
        self.gt_size=opt['gt_size']
        self.transform = transforms.Compose([transforms.Resize(self.gt_size), transforms.CenterCrop(self.gt_size), transforms.ToTensor()])

    def __getitem__(self, index):
        gt_path = self.paths['gt'][index]
        lq_path = self.paths['lq'][index]
        img_lq=self.transform(Image.open(lq_path).convert('RGB'))
        img_gt=self.transform(Image.open(gt_path).convert('RGB'))

        return {'lq': img_lq, 'gt': img_gt}

    def __len__(self):
        return len(self.paths['lq'])

@DATASET_REGISTRY.register()
class ImageMask_Pair_Loader(Image_Pair_Loader):
    def __init__(self, opt):
        Image_Pair_Loader.__init__(self,opt)
        self.opt = opt
        self.gt_folder, self.lq_folder,self.mask_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_mask']
        self.paths = glod_from_folder([self.lq_folder, self.gt_folder,self.mask_folder], ['lq', 'gt','mask'])
        self.to_tensor=transforms.ToTensor()
        self.gt_size=opt['gt_size']
        self.transform = transforms.Compose([transforms.Resize(self.gt_size), transforms.CenterCrop(self.gt_size), transforms.ToTensor()])

    def __getitem__(self, index):
        gt_path = self.paths['gt'][index]
        lq_path = self.paths['lq'][index]
        mask_path = self.paths['mask'][index]
        img_lq=self.transform(Image.open(lq_path).convert('RGB'))
        img_gt=self.transform(Image.open(gt_path).convert('RGB'))
        img_mask = self.transform(Image.open(mask_path).convert('RGB'))

        return {'lq': img_lq, 'gt': img_gt,'mask':img_mask}

    def __len__(self):
        return len(self.paths['lq'])
