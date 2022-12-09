import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)



import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image

from captum.attr import visualization as viz

import requests


image_lists = ['10008_airplane.png', '10005_cat.png', '10006_deer.png', '10007_frog.png']
for img_name in image_lists:
    
    img_file = "test_serve/image/"+img_name
    res = requests.post("http://localhost:8080/predictions/cifar", files={'data': open(img_file, 'rb')})

    print(res.json())


    res = requests.post("http://localhost:8080/explanations/cifar", files={'data': open(img_file, 'rb')})
    print(res.json())