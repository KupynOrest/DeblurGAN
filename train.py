import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from multiprocessing import freeze_support

def train(opt, data_loader, model, visualizer):
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	total_steps = 0
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
				print('PSNR on Train = %f' % psnrMetric)
				visualizer.display_current_results(results, epoch)

			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				if opt.display_id > 0:
					visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				model.save('latest')

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save('latest')
			model.save(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate()


if __name__ == '__main__':
	freeze_support()

	# python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)

	opt = TrainOptions().parse()
	opt.dataroot = 'D:\Photos\TrainingData\BlurredSharp\combined'
	opt.learn_residual = True
	opt.resize_or_crop = "crop"
	opt.fineSize = 256
	opt.gan_type = "gan"
	# opt.which_model_netG = "unet_256"

	# default = 5000
	opt.save_latest_freq = 100

	# default = 100
	opt.print_freq = 20

	data_loader = CreateDataLoader(opt)
	model = create_model(opt)
	visualizer = Visualizer(opt)
	train(opt, data_loader, model, visualizer)
