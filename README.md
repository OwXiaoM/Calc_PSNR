# Calculation of psnr between two pictures
from https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py

apply on the saving format of https://github.com/sanghyun-son/EDSR-PyTorch
```bash
python psnr_only.py --task classical_sr --scale 4 --folder_gt  {benchmark_path}/benchmark/{benchmark_name}/HR --folder_sr {result_save_path}/results-{benchmark_name}
```
