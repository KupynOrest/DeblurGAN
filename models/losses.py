import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd
import numpy as np
import torchvision.models as models
###############################################################################
# Functions
###############################################################################

class ContentLoss():
	
	def contentFunc():
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				print(layer)
				break
		return model

			
	def get_loss(fakeIm, realIm, contentFunc, criterion):
		f_fake = contentFunc.forward(fakeIm)
		f_real = contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = criterion(f_fake, f_real_no_grad)
		return loss

class DiscLoss()
    def name(self):
        return 'DiscLoss'

    def initialize(self, opt):
		self.criterionGAN = nn.BCECriterion()
		self.fake_AB_pool = ImagePool(opt.pool_size)
		
	def get_g_loss(self,net, realA, fakeB):
		# First, G(A) should fake the discriminator
		fake_AB = torch.cat((realA,fakeB), 1)
		pred_fake = net.forward(fake_AB)
		return self.criterionGAN(pred_fake, True)
		
    def get_loss(self, net, realA, fakeB, realB):
        # Fake
		# stop backprop to the generator by detaching fake_B
		fake_AB = self.fake_AB_pool.query(torch.cat((realA, fakeB), 1))
		self.pred_fake = net.forward(fake_AB.detach())
		self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

		# Real
		real_AB = torch.cat((realA, realB), 1)
		self.pred_real = net.forward(real_AB)
		self.loss_D_real = self.criterionGAN(self.pred_real, True)

		# Combined loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		return self.loss_D
		
class DiscLossLS(DiscLoss)
    def name(self):
        return 'DiscLossLS'

    def initialize(self, opt):
		DiscLoss.initialize(self, opt)
		self.criterionGAN = nn.L1Loss()
		
	def get_g_loss(self,net, realA, fakeB):
		return DiscLoss.get_g_loss(self,net, realA, fakeB)
		
    def get_loss(self, net, real, fake):
		return DiscLoss.get_loss(self, net, real, fake)
		
class DiscLossWGAN-GP(DiscLossLS)
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt):
		DiscLossLS.initialize(self, opt)
		self.LAMBDA = 10
		
	def get_g_loss(self,net, realA, fakeB):
		return DiscLossLS.get_g_loss(self,net, realA, fakeB)
		
	def calc_gradient_penalty(netD, real_data, fake_data):
    	alpha = torch.rand(1, 1)
    	alpha = alpha.expand(real_data.size())
	    alpha = alpha.cuda()

	    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	    interpolates = interpolates.cuda()
	    interpolates = autograd.Variable(interpolates, requires_grad=True)

	    disc_interpolates = netD(interpolates)

	    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
	                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
	                              create_graph=True, retain_graph=True, only_inputs=True)[0]

	    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
	    return gradient_penalty
		
    def get_loss(self, net, real, fake):
        lossD = DiscLossLS.get_loss(self, net, real, fake)
		return lossD + calc_gradient_penalty(net, real, fake)


def init_loss(opt):
	loss = None
	if opt.gan_type == 'wgan-gp':
		loss = DiscLossWGAN-GP()
	elif if opt.gan_type == 'lsgan':
		loss = DiscLossLS()
	elif opt.gan_type == 'gan':
		loss = DiscLoss()
	else:
		raise ValueError("GAN [%s] not recognized." % opt.gan_type)
	loss.initialize(opt)
	return loss