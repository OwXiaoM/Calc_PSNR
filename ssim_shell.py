import argparse
import subprocess
import time
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default=None)  
parser.add_argument('--scale', type=str, default=2, help='scale factor: 1, 2, 3, 4, 8')  
args = parser.parse_args()
exp_name = args.exp_name
benchmarks =['Set5','Set14','B100','Urban100','Manga109']
scale = args.scale
for b in benchmarks:
    p = subprocess.Popen('python3 psnr_multi.py --task classical_sr --scale {} --folder_gt ../data/benchmark/{}/HR --folder_sr ../experiment/{}/results-{}'.format(scale,b,exp_name,b),
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        encoding='utf-8'
                        )
    w = p.communicate()[0]
    print(w)
    with open('../experiment/{}/ssim.txt'.format(exp_name),'a') as f:
        f.write(w)

end = time.time()
print('All takes {} s'.format(str(end-start)))
