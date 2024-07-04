# PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts

- The 18th European Conference on Computer Vision ECCV 2024

[Zewen Chen](https://zwchen.top/), Haina Qin, [Jun Wang](), **[Chunfeng Yuan]()**, [Bing Li]()

National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences (NLPR, CASIA).

---

:rocket:  :rocket: :rocket: **News:**
- To be updated...
- ✅ **March, 2023**: We created this repository.

[![paper](https://img.shields.io/badge/arXiv-Paper-green.svg)]()
[![download](https://img.shields.io/github/downloads/chencn2020/PromptIQA/total.svg)](https://github.com/chencn2020/PromptIQA/releases)
[![Open issue](https://img.shields.io/github/issues/chencn2020/PromptIQA)](https://github.com/chencn2020/PromptIQA/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/chencn2020/PromptIQA)](https://github.com/chencn2020/PromptIQA/issues)
![visitors](https://visitor-badge.glitch.me/badge?page_id=chencn2020/PromptIQA)
[![GitHub Stars](https://img.shields.io/github/stars/chencn2020/PromptIQA?style=social)](https://github.com/chencn2020/PromptIQA)

[Online Demo[TBU]]()

## Checklist

- [] Code for PromptIQA
- [] Code for training
- [] Code for testing
- [] Checkpoint
- [] [Online Demo]() on huggingface

## Catalogue
1. [Introduction](#Introduction)
2. [Demonstrate](#Demonstrate)
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
    src="./image/framework/framework.svg" width="100%" alt=""/>
</div>
<div style="font-size: large; text-align: center;">
    <p>Figure1: The framework of the proposed PromptIQA.</p>
</div>



## Demonstrate
<div id="Demonstrate"></div>


## Usage For Training
<div id="Training"></div>

### Preparation

The dependencies for this work as follows:

```commandline
TBU
```

You can also run the following command to install the environment directly:

```commandline
TBU
```

---

You can download the LIVEC, BID, SPAQ and KonIQ datasets from the following download link. (TBU)

|        Dataset        |                        Image Number                        |  Score Type  |                                  Download Link                                  |
|:---------------------:|:----------------------------------------------------------:|:------------:|:-------------------------------------------------------------------------------:|
|         LIVEC         |     1162 images taken on a variety of mobile devices.      |     MOS      |       <a href="https://live.ece.utexas.edu/research/ChallengeDB/index.html" target="_blank">Link</a>       |
|          BID          |                   586 real-blur images.                    |     MOS      | <a href="https://github.com/zwx8981/UNIQUE#link-to-download-the-bid-dataset" target="_blank">Link</a>      |
|         SPAQ          |          11,125 images from 66 smartphone images.          |     MOS      |                     <a href="https://github.com/h4nwei/SPAQ" target="_blank">Link</a>                      |
|         KonIQ         |  10,073 images selected from public multimedia resources.  |     MOS      |           <a href="http://database.mmsp-kn.de/koniq-10k-database.html" target="_blank">Link</a>           |



### Training process

1. You should replace the dataset path in [dataset_info.json](./utils/dataset/dataset_info.json) to your own dataset path.
2. Run the following command to train the PromptIQA (Please review the [train.py](train.py) for more options).
```commandline
TBU
```


## Usage For Testing
<div id="Inference"> </div>



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


## Acknowledgement
<div id="Acknowledgement"></div>

We sincerely thank the great work [HyperIQA](https://github.com/SSL92/hyperIQA), [MANIQA](https://github.com/IIGROUP/MANIQA) and [MoCo](https://github.com/facebookresearch/moco). 
The code structure is partly based on their open repositories.