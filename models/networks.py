import torch
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import numpy as np
import sys
from torch.autograd import Function
import math
import h5py
from skimage.transform import resize

###############################################################################
# Functions
###############################################################################


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def get_norm_layer(norm_type):
	if norm_type == 'batch':
		norm_layer = nn.BatchNorm2d
	elif norm_type == 'instance':
		norm_layer = nn.InstanceNorm2d
	else:
		print('normalization layer [%s] is not found' % norm)
	return norm_layer

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
	netG = None
	use_gpu = len(gpu_ids) > 0
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if which_model_netG == 'resnet_9blocks':
		netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
	elif which_model_netG == 'resnet_6blocks':
		netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
	elif which_model_netG == 'unet_128':
		netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
	elif which_model_netG == 'unet_256':
		netG = MultiUnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
		netG = UnetWrapper(netG)
	else:
		print('Generator model name [%s] is not recognized' % which_model_netG)
	
	if len(gpu_ids) > 0:
		netG.cuda(gpu_ids[0])

	netG.apply(weights_init)
	return netG


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


class Sparse(Function):
	# Sparse matrix for S
	def forward(self, input, S):
		self.save_for_backward(S)
		output = torch.mm(S, input)
		# output = output.cuda()
		return output

	# This function has only a single output, so it gets only one gradient
	def backward(self, grad_output):
		S,  = self.saved_tensors
		grad_weight  = None
		grad_input = torch.mm(S.t(), grad_output)
		return grad_input, grad_weight

class JointLoss(nn.Module):
	def __init__(self):
		super(JointLoss, self).__init__()
		self.w_rc = 2.0 
		self.w_reconstr = 1.0
		self.w_rs_dense = 4.0
		self.w_ss = 1.0
		self.w_cy = 2.0
		self.w_regularization = 0.1
		self.local_s_w = np.array([[0.5,    0.5,   0.5,    0.5,    0.5], \
								   [0.5,    1 ,    1 ,     1,      0.5],\
								   [0.5,    1,     1,      1,      0.5],\
								   [0.5,    1,     1,      1,      0.5],\
								   [0.5,    0.5,   0.5,    0.5,    0.5]])
		x = np.arange(-1, 2)
		y = np.arange(-1, 2)
		self.X, self.Y = np.meshgrid(x, y)
		self.total_loss = None
		self.running_stage = 0

	# reflectance consistency loss, all maps are in log domain
	def RefConsistencyLoss(self, R_1, R_2,targets):
		mask = targets['mask'].cuda()
		num_val_pixels = torch.nonzero(mask).size(0)
		loss = torch.mul(Variable(mask, requires_grad = False), torch.pow(R_1 - R_2, 2))/num_val_pixels
		return torch.sum(loss)


	def MultiViewRefConsistencyLoss(self, R, targets):
		#  3XHXW
		M = Variable(targets['mask_0'].cuda(), requires_grad = False)
		M = M[:,0,:,:].unsqueeze(1).repeat(1, R.size(1), 1,1)

		assert(M.size(1) == R.size(1))

		sum_M = torch.sum(M, 0)

		Z = torch.sum(torch.mul(sum_M,sum_M) - sum_M)
		# N*3*HXW
		sum_MR2 = torch.sum(torch.mul(M, torch.pow(R,2)),0)
		sum_MR = torch.sum(torch.mul(M, R),0)
		sum_MR_2 = torch.mul(sum_MR,sum_MR)

		return torch.sum( (torch.mul(sum_M, sum_MR2) - sum_MR_2)/Z )

	def MultiViewRelightingLoss(self, input_img, log_R, log_S, targets):
		# rgb_img = input_img
		rgb_img = Variable(targets['rgb_img'].cuda(), requires_grad = False)
		log_img = torch.log(rgb_img)
		
		if log_R.size(1) == 1:
			print("One channel R")
			chromaticity = rgb_img/ torch.sum(rgb_img,1).unsqueeze(1).repeat(1,3,1,1)
			R = torch.log(torch.mul(chromaticity, torch.exp(log_R).repeat(1,3,1,1)))
		else:
			print("Three channel R")
			R = log_R

		S = log_S

		M  = Variable(targets['mask_0'].cuda(), requires_grad = False)
		M = M[:,0,:,:].unsqueeze(1).repeat(1, rgb_img.size(1), 1,1)
		L  = torch.mean(rgb_img, 1).unsqueeze(1).repeat(1, rgb_img.size(1), 1,1)

		assert(L.size(1) == M.size(1))
		assert(L.size(1) == R.size(1))

		# P = M
		P = torch.mul( torch.pow(L, 0.125), M)
		Q = M
		Y = R
		X = log_img - S

		sum_M = torch.sum(M, 0)

		sum_Q2 = sum_M
		sum_P2X2 = torch.sum( torch.pow(torch.mul(P, X), 2),0)
		sum_P2 = torch.sum( torch.pow(P, 2),0)
		sum_Q2Y2 = torch.sum( torch.pow(torch.mul(Q, Y), 2),0)
		sum_P2X = torch.sum( torch.mul( torch.pow(P, 2), X),0)
		sum_Q2Y = torch.sum( torch.mul( torch.pow(Q, 2), Y),0)

		Z = torch.sum(torch.mul(sum_M, sum_M))      

		joint_loss = torch.mul(sum_Q2, sum_P2X2) + torch.mul(sum_P2, sum_Q2Y2) - 2* torch.mul(sum_P2X, sum_Q2Y)

		return torch.sum( joint_loss/Z)


	# This is multiview lienar time smoothness term
	def SpatialTemporalBilateralRefSmoothnessLoss(self, R1 ,targets, att, num_features):
		total_loss = Variable(torch.cuda.FloatTensor(1))
		total_loss[0] = 0
		N = R1.size(2) * R1.size(3)
		Z = R1.size(1) * N * R1.size(0)
		num_c = R1.size(1)

		# we only have one sparse matrix 
		B_mat = targets[att+'B_list']
		S_mat = Variable(targets[att + 'S'].cuda(), requires_grad = False) # Splat and Slicing matrix
		n_vec = Variable(targets[att + 'N'].cuda(), requires_grad = False) # bi-stochatistic vector, which is diagonal matrix

		p = None
		# create a very long vector 
		for i in range(R1.size(0)):
			p1 = R1[i,:,:,:].view(num_c,-1).t() # NX3
			if i == 0:
				p = p1
			else:
				p = torch.cat((p, p1),0)

		p_norm_sum = torch.sum(torch.mul(p,p))

		Snp = torch.mul(n_vec.repeat(1,num_c), p)
		sp_mm = Sparse()
		Snp = sp_mm(Snp, S_mat)

		Snp_1 = Snp.clone()
		Snp_2 = Snp.clone()

		# # blur
		for f in range(num_features+1):
			B_var1 = Variable(B_mat[f].cuda(), requires_grad = False)
			sp_mm1 = Sparse()
			Snp_1 = sp_mm1(Snp_1, B_var1)
			
			B_var2 = Variable(B_mat[num_features-f].cuda(), requires_grad = False)               
			sp_mm2 = Sparse()
			Snp_2 = sp_mm2(Snp_2, B_var2)

		Snp_12 = Snp_1 + Snp_2
		pAp = torch.sum(torch.mul(Snp, Snp_12))
		total_loss = (p_norm_sum - pAp)/Z
		return total_loss

	def ShadingPenaltyLoss(self, S):
		return torch.mean(torch.abs(S - math.log(1.0)))

	def ReconstLoss(self, img, R, S, targets):
		log_img = torch.log(img) 
		sky_mask = Variable(targets['sky_mask'].cuda(), requires_grad = False)    
		luminance  = ( Variable(targets['L'].cuda(), requires_grad = False))
		# print(sky_mask.size())
		# print(luminance.size())
		weights = torch.mul(sky_mask, luminance)
		# weights = luminance
		num_val_pixels = torch.sum(sky_mask) 
		return torch.sum( torch.mul(weights, torch.pow(log_img - (R + S), 2) )/num_val_pixels )


	def RefSmoothnessLoss(self, R, targets):
		h = R.size(2)
		w = R.size(3)
		#  #img , #channel, h, w
		R_N = Variable(torch.cuda.FloatTensor( R.size(0), 9, R.size(1), R.size(2)-2, R.size(3)-2))
		total_loss = Variable(torch.cuda.FloatTensor(1))
		total_loss[0] = 0

		for i in range(0,9):
			R_N[:,i,:,:,:] = R[:,:,self.h_offset[i]:h-2 + self.h_offset[i], self.w_offset[i]:w-2+ self.w_offset[i] ] 

		for i in range(0, 8):
			vv = targets["w_r"][:,i,:,:].unsqueeze(1).repeat(1,3,1,1).cuda()
			r_diff = torch.mul(Variable(vv, requires_grad = False), torch.abs(R_N[:,8,:,:,:] - R_N[:,i,:,:,:])  ) 
			total_loss  = total_loss + torch.mean(r_diff)

		return total_loss/8.0

	def BilateralRefSmoothnessLoss(self, pred_R, targets, att, num_features):
		# pred_R = pred_R.cpu()
		total_loss = Variable(torch.cuda.FloatTensor(1))
		total_loss[0] = 0
		N = pred_R.size(2) * pred_R.size(3)
		Z = (pred_R.size(1) * N )

		# grad_input = torch.FloatTensor(pred_R.size())
		# grad_input = grad_input.zero_()

		for i in range(pred_R.size(0)): # for each image
			B_mat = targets[att+'B_list'][i] # still list of blur sparse matrices 
			S_mat = Variable(targets[att + 'S'][i].cuda(), requires_grad = False) # Splat and Slicing matrix
			n_vec = Variable(targets[att + 'N'][i].cuda(), requires_grad = False) # bi-stochatistic vector, which is diagonal matrix

			p = pred_R[i,:,:,:].view(pred_R.size(1),-1).t() # NX3
			# p'p
			# p_norm = torch.mm(p.t(), p) 
			# p_norm_sum = torch.trace(p_norm)
			p_norm_sum = torch.sum(torch.mul(p,p))

			# S * N * p
			Snp = torch.mul(n_vec.repeat(1,pred_R.size(1)), p)
			sp_mm = Sparse()
			Snp = sp_mm(Snp, S_mat)

			Snp_1 = Snp.clone()
			Snp_2 = Snp.clone()

			# # blur
			for f in range(num_features+1):
				B_var1 = Variable(B_mat[f].cuda(), requires_grad = False)
				sp_mm1 = Sparse()
				Snp_1 = sp_mm1(Snp_1, B_var1)
				
				B_var2 = Variable(B_mat[num_features-f].cuda(), requires_grad = False)               
				sp_mm2 = Sparse()
				Snp_2 = sp_mm2(Snp_2, B_var2)

			Snp_12 = Snp_1 + Snp_2
			pAp = torch.sum(torch.mul(Snp, Snp_12))

			total_loss = total_loss + ((p_norm_sum - pAp)/Z)


		total_loss = total_loss/pred_R.size(0) 
		# average over all images
		return total_loss

	def LocalShadSmoothenessLoss(self, S, targets, scale_idx):
		h = S.size(2)
		w = S.size(3)
		half_window_size = 1
		total_loss = Variable(torch.cuda.FloatTensor(1))
		total_loss[0] = 0

		M = targets['mask_'+ str(scale_idx)].float().cuda()
		mask_center = M[:,:,half_window_size + self.Y[half_window_size,half_window_size]:h-half_window_size + self.Y[half_window_size,half_window_size], \
						 half_window_size + self.X[half_window_size,half_window_size]:w-half_window_size + self.X[half_window_size,half_window_size]]

		S_center = S[:,:,half_window_size + self.Y[half_window_size,half_window_size]:h-half_window_size + self.Y[half_window_size,half_window_size], \
						 half_window_size + self.X[half_window_size,half_window_size]:w-half_window_size + self.X[half_window_size,half_window_size] ]

		c_idx = 0
		for k in range(0,half_window_size*2+1):
			for l in range(0,half_window_size*2+1):
				shading_weights = targets["shading_w_"+str(scale_idx)][:,c_idx,:,:].unsqueeze(1).float().cuda()
				
				mask_N = M[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
				composed_M = torch.mul(mask_N, mask_center)
				shading_weights = torch.mul(shading_weights, composed_M)
				
				S_N = S[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
				s_diff = torch.mul( Variable(shading_weights, requires_grad = False), torch.pow(S_center - S_N,2)  )  
				N = torch.sum(composed_M)
				total_loss  = total_loss + torch.sum(s_diff)/N
				c_idx = c_idx + 1

		return total_loss/8.0

	def LocalAlebdoSmoothenessLoss(self, R, targets, scale_idx):
		h = R.size(2)
		w = R.size(3)
		half_window_size = 1
		total_loss = Variable(torch.cuda.FloatTensor(1))
		total_loss[0] = 0

		R_center = R[:,:,half_window_size + self.Y[half_window_size,half_window_size]:h-half_window_size + self.Y[half_window_size,half_window_size], \
						 half_window_size + self.X[half_window_size,half_window_size]:w-half_window_size + self.X[half_window_size,half_window_size] ]

		M = targets['mask_'+ str(scale_idx)].repeat(1,3,1,1).float().cuda()


		mask_center = M[:,:,half_window_size + self.Y[half_window_size,half_window_size]:h-half_window_size + self.Y[half_window_size,half_window_size], \
						 half_window_size + self.X[half_window_size,half_window_size]:w-half_window_size + self.X[half_window_size,half_window_size]]

		c_idx = 0

		for k in range(0,half_window_size*2+1):
			for l in range(0,half_window_size*2+1):
				albedo_weights = targets["r_w_s"+str(scale_idx)][:,c_idx,:,:].unsqueeze(1).repeat(1,3,1,1).float().cuda()
				R_N = R[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
				mask_N = M[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
				composed_M = torch.mul(mask_N, mask_center)
				albedo_weights = torch.mul(albedo_weights, composed_M)
				r_diff = torch.mul( Variable(albedo_weights, requires_grad = False), torch.abs(R_center - R_N)  )  

				N = torch.sum(composed_M)
				total_loss  = total_loss + torch.sum(r_diff)/N
				c_idx = c_idx + 1


		return total_loss/(8.0)

	def __call__(self, input_images, prediction_S, prediction_R, _rgb_s, targets, data_set_name, epoch):
		num_images = prediction_S.size(0) # must be even number 
		mid_point = num_images/2

		# undo gamma correction
		prediction_S = prediction_S / 0.4545
		prediction_R = prediction_R / 0.4545
		_rgb_s = _rgb_s / 0.4545
		
		rgb_s = _rgb_s.unsqueeze(2)
		rgb_s = rgb_s.unsqueeze(3)
		rgb_s = rgb_s.repeat(1,1, prediction_S.size(2), prediction_S.size(3))

		prediction_Sr = prediction_S.repeat(1,3,1,1)
		prediction_Sr = prediction_Sr + rgb_s

		# downsample using bilinear inteporlation
		# down_sample = nn.AvgPool2d(2, stride=2)
		# prediction_S_1 = down_sample(prediction_S)
		# prediction_S_2 = down_sample(prediction_S_1)
		# prediction_S_3 = down_sample(prediction_S_2)

		# downsample using NN
		prediction_S_1 = prediction_S[:,:,::2,::2]
		prediction_S_2 = prediction_S_1[:,:,::2,::2]
		prediction_S_3 = prediction_S_2[:,:,::2,::2]

		rc_loss =  self.w_rc * self.MultiViewRefConsistencyLoss(prediction_R,targets) 
		rl_loss =  self.w_cy * self.MultiViewRelightingLoss(input_images, prediction_R, prediction_Sr, targets) 

		# multi-scale shading smoothness term
		ss_loss =  self.w_ss  * self.LocalShadSmoothenessLoss(prediction_S, targets,0)
		ss_loss = ss_loss + 0.5 * self.w_ss * self.LocalShadSmoothenessLoss(prediction_S_1, targets,1)
		ss_loss = ss_loss + 0.3333 * self.w_ss  * self.LocalShadSmoothenessLoss(prediction_S_2, targets,2)
		ss_loss = ss_loss + 0.25 * self.w_ss   * self.LocalShadSmoothenessLoss(prediction_S_3, targets,3)

		# spatial-temporal densely connected smoothness term
		rs_loss =  self.w_rs_dense * self.SpatialTemporalBilateralRefSmoothnessLoss(prediction_R, targets, 'R' ,5) 
		shading_color_loss = self.w_regularization * self.ShadingPenaltyLoss(prediction_S)

		print("ss loss", ss_loss.data[0])
		print("rs_loss", rs_loss.data[0])
		print("rl_loss loss", rl_loss.data[0])
		print("rc_loss loss", rc_loss.data[0])
		print("regularization loss ", shading_color_loss.data[0])

		total_loss = ss_loss + rs_loss + rl_loss  + rc_loss + shading_color_loss 

		self.total_loss = total_loss

		return total_loss.data[0]

	def get_loss_var(self):
		return self.total_loss


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
		assert(n_blocks >= 0)
		super(ResnetGenerator, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.gpu_ids = gpu_ids

		model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
				 norm_layer(ngf, affine=True),
				 nn.ReLU(True)]

		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
								stride=2, padding=1),
					  norm_layer(ngf * mult * 2, affine=True),
					  nn.ReLU(True)]

		mult = 2**n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1),
					  norm_layer(int(ngf * mult / 2), affine=True),
					  nn.ReLU(True)]

		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
		model += [nn.Tanh()]

		self.model = nn.Sequential(*model)

	def forward(self, input):
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
		conv_block = []
		p = 0
		# TODO: support padding types
		assert(padding_type == 'zero')
		p = 1

		# TODO: InstanceNorm
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
					   norm_layer(dim, affine=True),
					   nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
					   norm_layer(dim, affine=True)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64,
				 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(UnetGenerator, self).__init__()
		self.gpu_ids = gpu_ids

		# currently support only input_nc == output_nc
		# assert(input_nc == output_nc)

		# construct unet structure
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
		for i in range(num_downs - 5):
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

		self.model = unet_block

	def forward(self, input):
		if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost

		downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
							 stride=2, padding=1)
		downrelu = nn.LeakyReLU(0.2, False)
		downnorm = norm_layer(inner_nc, affine=True)
		uprelu = nn.ReLU(False)
		upnorm = norm_layer(outer_nc, affine=True)

		if outermost:
			n_output_dim = 3
			uprelu1 = nn.ReLU(False)
			uprelu2 = nn.ReLU(False)
			upconv_1 = nn.ConvTranspose2d(inner_nc * 2, inner_nc,
										kernel_size=4, stride=2,
										padding=1)
			upconv_2 = nn.ConvTranspose2d(inner_nc * 2, inner_nc,
										kernel_size=4, stride=2,
										padding=1)

			conv_1 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
							 stride=1, padding=1)            
			conv_2 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
							 stride=1, padding=1)

			# conv_1_o = nn.Conv2d(inner_nc, 1, kernel_size=3,
							 # stride=1, padding=1)            
			conv_2_o = nn.Conv2d(inner_nc, n_output_dim, kernel_size=3,
							 stride=1, padding=1)

			upnorm_1 = norm_layer(inner_nc, affine=True)
			upnorm_2 = norm_layer(inner_nc, affine=True)
			# uprelu2_o = nn.ReLU(False)

			down = [downconv]
			up_1 = [uprelu1, upconv_1, upnorm_1, nn.ReLU(False), conv_1, nn.ReLU(False), conv_1_o]
			up_2 = [uprelu2, upconv_2, upnorm_2, nn.ReLU(False), conv_2, nn.ReLU(False), conv_2_o]

			self.downconv_model = nn.Sequential(*down)
			self.upconv_model_1 = nn.Sequential(*up_1)
			self.upconv_model_2 = nn.Sequential(*up_2)
			self.submodule = submodule

		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
			self.model = nn.Sequential(*model)

		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up
			self.model = nn.Sequential(*model)

		# self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			# return self.model(x)
			down_x = self.downconv_model(x)
			y = self.submodule.forward(down_x)
			y_1 = self.upconv_model_1(y)
			y_2 = self.upconv_model_2(y)

			return y_1, y_2

		else:
			return torch.cat([self.model(x), x], 1)


class SingleUnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64,
				 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(SingleUnetGenerator, self).__init__()
		self.gpu_ids = gpu_ids

		# currently support only input_nc == output_nc
		# assert(input_nc == output_nc)

		# construct unet structure
		unet_block = SingleUnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
		for i in range(num_downs - 5):
			unet_block = SingleUnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = SingleUnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
		unet_block = SingleUnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
		unet_block = SingleUnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
		unet_block = SingleUnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

		self.model = unet_block

	def forward(self, input):
		if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)

class SingleUnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		super(SingleUnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost

		downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
							 stride=2, padding=1)
		downrelu = nn.LeakyReLU(0.2, False)
		downnorm = norm_layer(inner_nc, affine=True)
		uprelu = nn.ReLU(False)
		upnorm = norm_layer(outer_nc, affine=True)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, 3,
										kernel_size=4, stride=2,
										padding=1)

			down = [downconv]
			up = [uprelu, upconv]
			model = down + [submodule] + up
			self.model = nn.Sequential(*model)
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
			self.model = nn.Sequential(*model)

		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up
			self.model = nn.Sequential(*model)

		# self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([self.model(x), x], 1)


class MultiUnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64,
				 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(MultiUnetGenerator, self).__init__()
		self.gpu_ids = gpu_ids

		# currently support only input_nc == output_nc
		# assert(input_nc == output_nc)

		# construct unet structure
		unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
		for i in range(num_downs - 5):
			unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = MultiUnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
		unet_block = MultiUnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
		unet_block = MultiUnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)

		unet_block = MultiUnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

		self.model = unet_block

	def forward(self, input):
		if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)
			# self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|

class MultiUnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64,
				 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(MultiUnetGenerator, self).__init__()
		self.gpu_ids = gpu_ids

		# currently support only input_nc == output_nc
		# assert(input_nc == output_nc)

		# construct unet structure
		unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
		for i in range(num_downs - 5):
			unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = MultiUnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
		unet_block = MultiUnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
		unet_block = MultiUnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
		unet_block = MultiUnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

		self.model = unet_block

	def forward(self, input):
		if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)
			# self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class MultiUnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		super(MultiUnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		self.innermost = innermost
		# print("we are in mutilUnet")
		downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
							 stride=2, padding=1)
		downrelu = nn.LeakyReLU(0.2, False)
		downnorm = norm_layer(inner_nc, affine=True)
		uprelu = nn.ReLU(False)
		upnorm = norm_layer(outer_nc, affine=True)

		if outermost:
			n_output_dim = 3

			down = [downconv]

			upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
										kernel_size=4, stride=2,
										padding=1)]
			upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 3,
										kernel_size=4, stride=2,
										padding=1)]

		elif innermost:

			down = [downrelu, downconv]
			upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1), norm_layer(outer_nc, affine=True)]
			upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1), norm_layer(outer_nc, affine=True)]
			#  for rgb shading 
			int_conv = [nn.AdaptiveAvgPool2d((1,1)) , nn.ReLU(False),  nn.Conv2d(inner_nc, inner_nc/2, kernel_size=3, stride=1, padding=1), nn.ReLU(False)]
			fc = [nn.Linear(256, 3)]
			self.int_conv = nn.Sequential(* int_conv) 
			self.fc = nn.Sequential(* fc)
		else:

			down = [downrelu, downconv, downnorm]
			up_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1), norm_layer(outer_nc, affine=True)]
			up_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1), norm_layer(outer_nc, affine=True)]

			if use_dropout:
				upconv_model_1 = up_1 + [nn.Dropout(0.5)]
				upconv_model_2 = up_2 + [nn.Dropout(0.5)]
			else:
				upconv_model_1 = up_1
				upconv_model_2 = up_2
		

		self.downconv_model = nn.Sequential(*down)
		self.submodule = submodule
		self.upconv_model_1 = nn.Sequential(*upconv_model_1)
		self.upconv_model_2 = nn.Sequential(*upconv_model_2)

	def forward(self, x):

		if self.outermost:
			down_x = self.downconv_model(x)

			y_1, y_2, color_s = self.submodule.forward(down_x)
			y_1 = self.upconv_model_1(y_1)
			y_2 = self.upconv_model_2(y_2)

			return y_1, y_2, color_s

		elif self.innermost:
			down_output = self.downconv_model(x)
			color_s = self.int_conv(down_output)
			color_s = color_s.view(color_s.size(0), -1)
			color_s  = self.fc(color_s)

			y_1 = self.upconv_model_1(down_output)
			y_2 = self.upconv_model_2(down_output)  
			y_1 = torch.cat([y_1, x], 1)
			y_2 = torch.cat([y_2, x], 1)


			return y_1, y_2, color_s
		else:
			down_x = self.downconv_model(x)
			y_1, y_2, color_s = self.submodule.forward(down_x)
			y_1 = self.upconv_model_1(y_1)
			y_2 = self.upconv_model_2(y_2)
			y_1 = torch.cat([y_1, x], 1)
			y_2 = torch.cat([y_2, x], 1)

			return y_1, y_2, color_s

class UnetWrapper(nn.Module):
	def __init__(self, model):
		super(UnetWrapper, self).__init__()
		self.model = model

	def forward(self, input_):
		prediction_S, prediction_R, rgb_s = self.model(input_)
		# gamma correction
		prediction_S = prediction_S * 0.4545
		prediction_R = prediction_R * 0.4545
		rgb_s = rgb_s * 0.4545

		return prediction_S, prediction_R, rgb_s