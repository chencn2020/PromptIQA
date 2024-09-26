import os
import random
import torchvision
import cv2
import torch
from PromptIQA.models import promptiqa
import numpy as np
from PromptIQA.utils.dataset.process import ToTensor, Normalize
from PromptIQA.utils.toolkit import *
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append(os.path.dirname(__file__))


def load_model(pkl_path):
    model = promptiqa.PromptIQA()
    dict_pkl = {}
    for key, value in torch.load(pkl_path, map_location="cpu")["state_dict"].items():
        dict_pkl[key[7:]] = value
    model.load_state_dict(dict_pkl)
    print("Load Model From ", pkl_path)
    return model


class PromptIQA:
    def __init__(self, pkl_path="./PromptIQA/checkpoints/best_model.pth.tar") -> None:
        self.pkl_path = pkl_path
        self.model = load_model(self.pkl_path).cuda()
        self.model.eval()

        self.transform = torchvision.transforms.Compose(
            [Normalize(0.5, 0.5), ToTensor()]
        )

    def get_an_img_score(self, img_path, target=0):
        def load_image(img_path, size=224):
            if isinstance(img_path, str):
                d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else:
                d_img = img_path
            d_img = cv2.resize(d_img, (size, size), interpolation=cv2.INTER_CUBIC)
            d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
            d_img = np.array(d_img).astype("float32") / 255
            d_img = np.transpose(d_img, (2, 0, 1))
            return d_img

        sample = load_image(img_path)
        samples = {"img": sample, "gt": target}
        samples = self.transform(samples)

        return samples

    def run(self, ISPP_I, ISPP_S, image):
        img_tensor, gt_tensor = None, None

        for isp_i, isp_s in zip(ISPP_I, ISPP_S):
            score = np.array(isp_s)
            samples = self.get_an_img_score(isp_i, score)

            if img_tensor is None:
                img_tensor = samples["img"].unsqueeze(0)
                gt_tensor = samples["gt"].type(torch.FloatTensor).unsqueeze(0)
            else:
                img_tensor = torch.cat((img_tensor, samples["img"].unsqueeze(0)), dim=0)
                gt_tensor = torch.cat(
                    (gt_tensor, samples["gt"].type(torch.FloatTensor).unsqueeze(0)),
                    dim=0,
                )

        img = img_tensor.squeeze(0).cuda()
        label = gt_tensor.squeeze(0).cuda()
        self.model.forward_prompt(img, label.reshape(-1, 1), "example")

        samples = self.get_an_img_score(image)
        img = samples["img"].unsqueeze(0).cuda()
        pred = self.model.inference(img, "example")

        return round(pred.item(), 4)


if __name__ == "__main__":
    promptIQA = PromptIQA()

    ISPP_I = [
        "./Examples/Example1/ISPP/1600.AWGN.1.png",
        "./Examples/Example1/ISPP/1600.AWGN.2.png",
        "./Examples/Example1/ISPP/1600.AWGN.3.png",
        "./Examples/Example1/ISPP/1600.AWGN.4.png",
        "./Examples/Example1/ISPP/1600.AWGN.5.png",
        "./Examples/Example1/ISPP/1600.BLUR.1.png",
        "./Examples/Example1/ISPP/1600.BLUR.2.png",
        "./Examples/Example1/ISPP/1600.BLUR.3.png",
        "./Examples/Example1/ISPP/1600.BLUR.4.png",
        "./Examples/Example1/ISPP/1600.BLUR.5.png",
    ]

    ISPP_S = [0.062, 0.206, 0.262, 0.375, 0.467, 0.043, 0.142, 0.341, 0.471, 0.75]
    Image = "./Examples/Example1/cactus.png"

    score = promptIQA.run(ISPP_I, ISPP_S, Image)
