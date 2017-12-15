import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
	return window

def SSIM(img1, img2):
	(_, channel, _, _) = img1.size()
	window_size = 11
	window = create_window(window_size, channel)
	mu1 = F.conv2d(img1, window, padding = window_size/2, groups = channel)
	mu2 = F.conv2d(img2, window, padding = window_size/2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size/2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size/2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size/2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()
	
def PSNR(img1, img2):
	mse = np.mean( (img1/255. - img2/255.) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
