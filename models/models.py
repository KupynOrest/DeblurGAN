from .conditional_gan_model import ConditionalGAN

def create_model(opt):
	model = ConditionalGAN()
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
