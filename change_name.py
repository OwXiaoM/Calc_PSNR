import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str,default =None,help='input file path')
args = parser.parse_args()
file_path = args.f
file_path = r'/home/xiaomimg/edsr/EDSR-PyTorch/experiment/rcanx2_img/results-Set5'

for name in os.listdir(file_path):
    new_name = name.replace('_x2_SR','')
    os.rename(file_path+'/'+name,file_path+'/'+new_name)
