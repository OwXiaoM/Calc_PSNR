import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch

from utils import util_calculate_psnr_ssim as util

#--task classical_sr --scale 4 --folder_gt  testsets/div2k/HR --folder_sr results/ntire_div2kx4
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--folder_sr', type=str, default=None, help='input sr image folder')
    args = parser.parse_args()

    # setup folder and path
    folder, border = setup(args)
    # print(border)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
 
        path_hr = args.folder_sr + path.replace(args.folder_gt, '')

        #imgname, img_gt, img_ht = get_image_pair(args, path, path_hr)  # image to HWC-BGR, float32
        #print('1',path)
        #print('2','/LR_bicubic/x{}'.format(str(args.scale)))
        path_lr =  (path.replace('/HR', '/LR_bicubic/X{}'.format(str(args.scale)))).replace('.png','x{}.png'.format(args.scale))
        #print('3',path_lr)
        imgname, img_gt, img_ht, img_lq = get_image_pair(args, path, path_hr, path_lr)  # image to HWC-BGR, float32
 
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to('cpu')  # CHW-RGB to NCHW-RGB

        _, _, h_old, w_old = img_lq.size()

        # save image
        # img_ht=torch.as_tensor(img_ht)
        output = img_ht
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
   
        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt   
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            if args.task in ['jpeg_car']:
                psnr_b = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_b'].append(psnr_b)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
                  'PSNR_B: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print(' \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format( ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))
        if args.task in ['jpeg_car']:
            ave_psnr_b = sum(test_results['psnr_b']) / len(test_results['psnr_b'])
            print('-- Average PSNR_B: {:.2f} dB'.format(ave_psnr_b))


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        folder = args.folder_gt
        border = args.scale


    return folder, border


def get_image_pair(args, path, path_hr, path_lr):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_hr = cv2.imread(path_hr, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = cv2.imread(path_lr, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_gt, img_hr, img_lq



if __name__ == '__main__':
    main()

