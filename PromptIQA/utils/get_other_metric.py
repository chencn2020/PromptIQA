import os
import numpy as np

path = '/disk1/chenzewen/OurIdeas/GIQA/GIQA_2024/Formal/Full_model'
file_path = 'Other_Metric_Inference/tid2013_other/inference_log_fix_0_SSIM.log'

for seed in os.listdir(path):
    file = os.path.join(path, seed, file_path)
    
    srocc, plcc = [], []
    with open(file, 'r') as f:
        info = f.readlines()[-1].split()
        print(info)
        srocc.append(float(info[-3][:-1]))
        plcc.append(float(info[-1]))

print(np.median(srocc))
print(np.median(plcc))