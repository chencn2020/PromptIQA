# PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts

- The 18th European Conference on Computer Vision ECCV 2024

---

:rocket:  :rocket: :rocket: **News:**
- To be updated...
- ✅ **February, 2025**: We pubulish the training code.
- ✅ **September, 2024**: We pubulish the checkpoints and testing code.
- ✅ **September, 2024**: We pubulish the online demo.
- ✅ **March, 2024**: We created this repository.

[![paper](https://img.shields.io/badge/Paper-green.svg)](https://link.springer.com/chapter/10.1007/978-3-031-73232-4_14)
[![download](https://img.shields.io/github/downloads/chencn2020/PromptIQA/total.svg)](https://github.com/chencn2020/PromptIQA/releases)
[![Open issue](https://img.shields.io/github/issues/chencn2020/PromptIQA)](https://github.com/chencn2020/PromptIQA/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/chencn2020/PromptIQA)](https://github.com/chencn2020/PromptIQA/issues)
[![Static Badge](https://img.shields.io/badge/Try_Demo-6B88E3?logo=youtubegaming&logoColor=DAE4EE)](https://huggingface.co/spaces/Zevin2023/PromptIQA) 
[![GitHub Stars](https://img.shields.io/github/stars/chencn2020/PromptIQA?style=social)](https://github.com/chencn2020/PromptIQA)


## Checklist

- [x] Code for training
- [x] Code for PromptIQA
- [x] Code for testing
- [x] Checkpoint
- [x] [Online Demo](https://huggingface.co/spaces/Zevin2023/PromptIQA) on huggingface

## Catalogue
1. [Introduction](#Introduction)
2. [Try Our Demo](#Demo)
3. [Usage For Training](#Training)
4. [Usage For Testing](#Testing)
5. [Results](#Results)
6. [Citation](#Citation)
7. [Acknowledgement](#Acknowledgement)


## Introduction
<div id="Introduction"></div>

This is an official implementation of **PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts** by Pytorch.

---

> Due to the diversity of assessment requirements in various application scenarios for the IQA task, existing IQA methods struggle to directly adapt to these varied requirements after training. Thus, when facing new requirements, a typical approach is fine-tuning these models on datasets specifically created for those requirements. However, it is time-consuming to establish IQA datasets. In this work, we propose a Prompt-based IQA (PromptIQA) that can directly adapt to new requirements without fine-tuning after training. On one hand, it utilizes a short sequence of Image-Score Pairs (ISP) as prompts for targeted predictions, which significantly reduces the dependency on the data requirements. On the other hand, PromptIQA is trained on a mixed dataset with two proposed data augmentation strategies to learn diverse requirements, thus enabling it to effectively adapt to new requirements. Experiments indicate that the PromptIQA outperforms SOTA methods with higher performance and better generalization. 

<div style="display: flex; justify-content: center;">
    <img style="border-radius: 0.3125em;" 
    src="./image/framework/framework.png" width="100%" alt=""/>
</div>
<div style="font-size: large; text-align: center;">
    <p>Figure1: The framework of the proposed PromptIQA.</p>
</div>

## Try Our Demo 🕹️
<div id="Demo"></div>

**Click** 👇 **to try our demo online.**

[**Huggingface**](https://huggingface.co/spaces/Zevin2023/PromptIQA)

# Training and Testing

## Preparation

Run the following script to create the environment:
```commandline
conda create -n promptiqa python=3.8 -y
```

The dependencies for this work as follows:

```commandline
einops==0.7.0
numpy==1.24.4
opencv_python==4.8.0.76
openpyxl==3.1.2
Pillow==10.0.0
scipy
timm==0.5.4
torch==2.0.1+cu118
torchvision==0.15.2+cu118
tqdm==4.66.1
gradio
```

You can also run the following command to install the environment directly:

```commandline
pip install -r requirements.txt
```


## Usage For Training
<div id="Training"></div>

### Getting the datasets
You can download the total 12 datasets from the following download link. (TBU)

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Task</th>
            <th>Ref. Image</th>
            <th>Dist. Types</th>
            <th>Image Number</th>
            <th>Label Type</th>
            <th>Range</th>
            <th>Download</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>LIVE</td>
            <td  rowspan="4">Synthetic Distortion <br>Nature IQA</td>
            <td>29</td>
            <td>5</td>
            <td>779</td>
            <td>DMOS</td>
            <td>[1, 100]</td>
            <td><a href="https://live.ece.utexas.edu/index.php" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>CSIQ</td>
            <td>30</td>
            <td>6</td>
            <td>866</td>
            <td>DMOS</td>
            <td>[0, 1]</td>
           <td><a href="https://s2.smu.edu/~eclarson/csiq.html" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>TID2013</td>
            <td>25</td>
            <td>24</td>
            <td>3,000</td>
            <td>MOS</td>
            <td>[0, 9]</td>
           <td><a href="https://www.ponomarenko.info/tid2013.htm" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>Kadid-10k</td>
            <td>81</td>
            <td>25</td>
            <td>10,125</td>
            <td>MOS</td>
            <td>[1, 5]</td>
           <td><a href="https://database.mmsp-kn.de/kadid-10k-database.html" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>BID</td>
            <td rowspan="4">Authentic Distortion <br>Nature IQA</td>
            <td>-</td>
            <td>-</td>
            <td>586</td>
            <td>MOS</td>
            <td>[0, 5]</td>
            <td><a href="https://github.com/zwx8981/UNIQUE#link-to-download-the-bid-dataset" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>SPAQ</td>
            <td>-</td>
            <td>-</td>
            <td>11,125</td>
            <td>MOS</td>
            <td>[0, 100]</td>
            <td> <a href="https://github.com/h4nwei/SPAQ" target="_blank">Link</a>    </td>
        </tr>
        <tr>
            <td>LIVEC</td>
            <td>-</td>
            <td>-</td>
            <td>1,162</td>
            <td>MOS</td>
            <td>[1, 100]</td>
            <td><a href="https://live.ece.utexas.edu/research/ChallengeDB/index.html" target="_blank">Link</a> </td>
        </tr>
        <tr>
            <td>KonIQ-10K</td>
            <td>-</td>
            <td>-</td>
            <td>10,073</td>
            <td>MOS</td>
            <td>[0, 100]</td>
            <td><a href="http://database.mmsp-kn.de/koniq-10k-database.html" target="_blank">Link</a> </td>
        </tr>
        <tr>
            <td>GFIQA20k</td>
            <td>Face IQA</td>
            <td>-</td>
            <td>-</td>
            <td>19,988</td>
            <td>MOS</td>
            <td>[0, 1]</td>
           <td><a href="https://database.mmsp-kn.de/gfiqa-20k-database.html" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>AGIQA3k</td>
            <td rowspan="2">AI generated IQA</td>
            <td>-</td>
            <td>-</td>
            <td>2,982</td>
            <td>MOS</td>
            <td>[0, 1]</td>
           <td><a href="https://github.com/lcysyzxdxc/AGIQA-3k-Database" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>AIGCIQA2023</td>
            <td>-</td>
            <td>-</td>
            <td>2,400</td>
            <td>MOS</td>
            <td>[0, 1]</td>
           <td><a href="https://github.com/wangjiarui153/aigciqa2023" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td>UWIQA</td>
            <td>Underwater IQA</td>
            <td>-</td>
            <td>-</td>
            <td>890</td>
            <td>MOS</td>
            <td>[0, 1]</td>
           <td><a href="" target="_blank">Link</a></td>
        </tr>
    </tbody>
</table>

### Training process

1. You should replace the dataset path in [dataset_info.json](./PromptIQA/utils/dataset/dataset_info.json) to your own dataset path.
2. Run the following command to train the PromptIQA (Please review the [train.py](train.py) for more options). 

    P.S.⚠️ Supported datasets can be found in [dataset_info.json](./PromptIQA/utils/dataset/dataset_info.json)

```commandline
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train.py --dist-url 'tcp://localhost:12754' \
    --dataset spaq csiq livec koniq10k kadid live \ 
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --batch_size 44 --epochs 50 --seed 2024 \
    --random_flipping_rate 0.1 --random_scale_rate 0.5 \
    --model promptiqa \
    --save_path ./Exp/PromptIQA_2024
```

You can directly run the following script:

```commandline
sh train.sh
```

The number of prompts is ```batch_size / GPU_Num - 1```


## Usage For Testing
<div id="Inference"> </div>

### Pre-training Weight

You can get our pretraining weight from [Huggingface.](https://huggingface.co/Zevin2023/PromptIQA/tree/main)

Then put the checkpoints in ```./PromptIQA/checkpoints```

### Running On The Demo

You can use the following command to run the test demo:

```commandline
python3 app.py
```

### Running Testing Code 

You can use the following command to run the testing code:

```commandline
python3 test.py
```

## Results
<div id="Results"> </div>

We achieved state-of-the-art performance on most IQA datasets simultaniously within one single model. 

More detailed results can be found in the [paper](). 

<div style="display: flex; justify-content: center;">
    <img style="border-radius: 0.3125em;" 
    src="./image/experiments/EXP1.jpg" width="100%" alt=""/>
</div>
<div style="font-size: large; text-align: center;">
    <p>Individual Dataset Comparison.</p>
</div>

## Citation
<div id="Citation"> </div>

If our work is useful to your research, we will be grateful for you to cite our paper:

```
@inproceedings{chen2024promptiqa,
  title={Promptiqa: Boosting the performance and generalization for no-reference image quality assessment via prompts},
  author={Chen, Zewen and Qin, Haina and Wang, Juan and Yuan, Chunfeng and Li, Bing and Hu, Weiming and Wang, Liang},
  booktitle={European Conference on Computer Vision},
  pages={247--264},
  year={2024},
  organization={Springer}
}
```

## Stars ⭐️

<a href="https://star-history.com/#chencn2020/PromptIQA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=chencn2020/PromptIQA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=chencn2020/PromptIQA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=chencn2020/PromptIQA&type=Date" />
 </picture>
</a>


## Acknowledgement
<div id="Acknowledgement"></div>

We sincerely thank the great work [HyperIQA](https://github.com/SSL92/hyperIQA), [MANIQA](https://github.com/IIGROUP/MANIQA) and [MoCo](https://github.com/facebookresearch/moco). 
The code structure is partly based on their open repositories.
