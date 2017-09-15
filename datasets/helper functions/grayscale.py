from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

# Helper script to create dataset for image colorization

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for images', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='output directory', type=str, default='../dataset/test_B')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

splits = os.listdir(args.fold_A)

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_list = os.listdir(img_fold_A)
	
    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_B = os.path.join(args.fold_B, sp)
    if not os.path.isdir(img_fold_B):
        os.makedirs(img_fold_B)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)

        if os.path.isfile(path_A):
            name_B = name_A
			
            path_B = os.path.join(img_fold_B, name_B)
            im_A = cv2.imread(path_A, 0)
            cv2.imwrite(path_B, im_A)
