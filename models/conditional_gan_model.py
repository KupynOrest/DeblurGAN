import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalGAN(BaseModel):
	def name(self):
		return 'ConditionalGANModel'

	def __init__(self, opt):
		super(ConditionalGAN, self).__init__(opt)
		self.isTrain = opt.isTrain
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

		# load/define networks
		# Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		use_parallel = not opt.gan_type == 'wgan-gp'
		print("Use Parallel = ", "True" if use_parallel else "False")
		self.netG = networks.define_G(
			opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
			not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual
		)
		if self.isTrain:
			use_sigmoid = opt.gan_type == 'gan'
			self.netD = networks.define_D(
				opt.output_nc, opt.ndf, opt.which_model_netD,
				opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel
			)
		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch)

		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam( self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
			self.optimizer_D = torch.optim.Adam( self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
												
			self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
			
			# define loss functions
			self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		if self.isTrain:
			networks.print_network(self.netD)
		print('-----------------------------------------------')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		inputA = input['A' if AtoB else 'B']
		inputB = input['B' if AtoB else 'A']
		self.input_A.resize_(inputA.size()).copy_(inputA)
		self.input_B.resize_(inputB.size()).copy_(inputB)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B)

	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D(self):
		self.loss_D = self.discLoss.get_loss(self.netD, self.real_A, self.fake_B, self.real_B)

		self.loss_D.backward(retain_graph=True)

	def backward_G(self):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B)
		# Second, G(A) = B
		self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A

		self.loss_G = self.loss_G_GAN + self.loss_G_Content

		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()

		for iter_d in xrange(self.criticUpdates):
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
							('G_L1', self.loss_G_Content.item()),
							('D_real+fake', self.loss_D.item())
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		fake_B = util.tensor2im(self.fake_B.data)
		real_B = util.tensor2im(self.real_B.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)
		self.save_network(self.netD, 'D', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
