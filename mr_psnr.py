# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:20:22 2021

@author: zxlation
"""
import os
import numpy as np
from pathlib import Path
from utils import calc_psnr, calc_ssim

import warnings
warnings.filterwarnings('ignore')

def list_file(dirname, postfix = ''):
    filelist = []
    files = os.listdir(dirname)
    for item in files:
        cur_path = str(Path(os.path.join(dirname,item)))
        if os.path.isfile(cur_path):
            if item.endswith(postfix):
                filelist.append([dirname, item])
        else:
            if os.path.isdir(cur_path):
                filelist.extend(list_file(cur_path, postfix))
    return filelist


def calc_psnr_ssim(pred_vol, real_vol):
    if not real_vol.shape == pred_vol.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if real_vol.dtype != pred_vol.dtype:
        raise ValueError("Inputs have mismatched dtype.")
    
    data_range = np.max(real_vol) - np.min(real_vol)
    psnr = calc_psnr(real_vol, pred_vol, data_range)
    ssim = calc_ssim(real_vol, pred_vol, gaussian_weights = True, data_range = data_range,
                     multichannel = False, use_sample_covariance = False, pad = 0)
    
    return psnr, ssim


def calc_metrics(input_arr, label_arr):
    num_items = len(input_arr)
    assert num_items == len(label_arr)
    
    psnr, ssim = 0, 0
    for i in range(num_items):
        inp = np.load(str(Path(os.path.join(*input_arr[i]))))
        lab = np.load(str(Path(os.path.join(*label_arr[i]))))
        cur_psnr, cur_ssim = calc_psnr_ssim(inp, lab)
        psnr += cur_psnr
        ssim += cur_ssim
    
    psnr /= num_items
    ssim /= num_items
    
    return psnr, ssim


def ret_print(works, degradations, modals, scales, EvalTable):
    for w in range(len(works)):
        work = works[w]
        print('='*20)
        print(work)
        print('='*20)
        for d in range(len(degradations)):
            degradation = degradations[d]
            print('-- [%s]:' % degradation)
            for m in range(len(modals)):
                modal = modals[m]
                print('\t%s:' % modal)
                for s in range(len(scales)):
                    scale = scales[s]
                    psnr = EvalTable[w, d, m, s, 0]
                    ssim = EvalTable[w, d, m, s, 1]
                    print('\t X%d: %.4f/%.4f' % (scale, psnr, ssim))
                    


def main(argv=None):
    apath = 'works/'
    spath = 'result/'
    
    works = ['SRCNN']
    degradations = ['bicubic']
    modals = ['PD']
    scales = [2]
    
    
    EvalTable = np.zeros([len(works), len(degradations), len(modals), len(scales), 2])
    for m in range(len(modals)):
        modal = modals[m]
        print('process %s...' % modal)
        gt_path = str(Path(os.path.join(apath, 'GT', modal)))
        gt_list = sorted(list_file(gt_path, '.npy'))
        
        for w in range(len(works)):
            work = works[w]
            print('-- [%s]:' % work)
            for d in range(len(degradations)):
                degradation = degradations[d]
                print('\t[%s]...' % degradation)
                for s in range(len(scales)):
                    scale = scales[s]
                    print('\t X%d...' % scale)
                    re_path = str(Path(os.path.join(apath, work, degradation, modal, 'X'+str(scale))))
                    re_list = sorted(list_file(re_path, '.npy'))
                    psnr, ssim = calc_metrics(re_list, gt_list)
                    EvalTable[w, d, m, s, 0] = psnr
                    EvalTable[w, d, m, s, 1] = ssim
                    
    re_path = str(Path(os.path.join(spath, 'result.npy')))
    np.save(re_path, EvalTable)
    
    ret_print(works, degradations, modals, scales, EvalTable)
    
    print('done.')


if __name__ == '__main__':
    main()
