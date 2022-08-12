# Calculation of Peak Signal-to-Noise Ratio between two pictures
The scrip is extracted from https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py

And it should be applied on the saving format of https://github.com/sanghyun-son/EDSR-PyTorch
```bash
python psnr_only.py --task classical_sr --scale 4 --folder_gt  {benchmark_path}/benchmark/{benchmark_name}/HR --folder_sr {result_save_path}/results-{benchmark_name}
```
 **And I found that when calculting the metrics, it applys on image one by one, which is extremely slow, so I add a multi-process edition on the origin script, so we have psnr_multi.py. By tring, I found when process pool is 20, it is the fastest for 5 benchmark datasets of super-resolution.**
```bash
python psnr_only.py --task classical_sr --scale 4 --folder_gt  {benchmark_path}/benchmark/{benchmark_name}/HR --folder_sr {result_save_path}/results-{benchmark_name}
```
**At last I think when running the script on different benchmark, I have to change the benchmark name everytime and I found it annoying, so I write another file that helps me execute this script, which is ssim_shell.py, it also writes logs in the file path of certain experiment.**

Just
```bash
python ssim_shell.py --exp_name {the saved experiment name} --scale {scale, scale is important when cropping the border}
```
