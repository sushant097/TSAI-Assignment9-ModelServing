## Importing Libraries
import os
import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)

import shap
import timm
import torch
import urllib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T


from PIL import Image
from captum.robust import PGD, FGSM
from matplotlib.colors import LinearSegmentedColormap
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from captum.attr import (
    DeepLift,
    Saliency,
    Occlusion,
    NoiseTunnel,
    GradientShap,
    IntegratedGradients,
    visualization as viz,
)

device = torch.device("cuda")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
for i in os.listdir('images'):
    # if i == 'i4.png':
    #     continue
    img = Image.open('images' + i)
    # print dimension of img
    print("i: ", i, " and size: ", img.size)

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)


    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    inv_transform= T.Compose([
        T.Normalize(
            mean = (-1 * np.array(mean) / np.array(std)).tolist(),
            std = (1 / np.array(std)).tolist()
        ),
    ])
    pgd = PGD(model, torch.nn.CrossEntropyLoss(reduction='none'), lower_bound=-1, upper_bound=1)  # construct the PGD attacker

    perturbed_image_pgd = pgd.perturb(inputs=img_tensor, radius=0.13, step_size=0.02,
                                    step_num=7, target=torch.tensor([282]).to(device), targeted=True)
    new_pred_pgd, score_pgd = get_prediction(model, perturbed_image_pgd)

    print("new prediction: ", new_pred_pgd, score_pgd)


    fig1 = plt.gcf()
    npimg = inv_transform(perturbed_image_pgd.cpu()).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % new_pred_pgd + " " + str(score_pgd))
    plt.show()
    fig1.savefig('output/pgd_image/' + i, dpi=100)