import timm
import urllib
import torch

import numpy as np

import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt


device = torch.device("cuda")


model = timm.create_model("resnet18", pretrained=True)
model.eval()
model = model.to(device)


# Download human-readable labels for ImageNet.
# get the classnames
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
)
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]



transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])


transform_normalize = T.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
)


img = Image.open('images/cat-1.jpg')

transformed_img = transform(img)

img_tensor = transform_normalize(transformed_img)
img_tensor = img_tensor.unsqueeze(0)


img_tensor = img_tensor.to(device)
output = model(img_tensor)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

pred_label_idx.squeeze_()
predicted_label = categories[pred_label_idx.item()]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')



# Import for captum

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open('cat.jpeg')

img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor.requires_grad = True
img_tensor = img_tensor.to(device)

img_tensor.requires_grad

saliency = Saliency(model)
grads = saliency.attribute(img_tensor, target=285)
grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

original_image = np.transpose((img_tensor.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

original_image.shape

_ = viz.visualize_image_attr(None, original_image,
                             method="original_image", title="Original Image")

_ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                             show_colorbar=True, title="Overlayed Gradient Magnitudes")


def attribute_image_features(algorithm, input, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=285,
                                              **kwargs
                                              )

    return tensor_attributions


ig = IntegratedGradients(model)
attr_ig, delta = attribute_image_features(ig, img_tensor, baselines=img_tensor * 0, return_convergence_delta=True)
attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
print('Approximation delta: ', abs(delta))

_ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all",
                             show_colorbar=True, title="Overlayed Integrated Gradients")

# ## Captum Model Robustness


from captum.robust import FGSM
from captum.robust import PGD

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open('cat.jpeg')

img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor.requires_grad = True
img_tensor = img_tensor.to(device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_transform = T.Compose([
    T.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist()
    ),
])


def get_prediction(model, image: torch.Tensor):
    model = model.to(device)
    img_tensor = image.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]

    return predicted_label, prediction_score.squeeze().item()

    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')


# Get original prediction
pred, score = get_prediction(model, img_tensor)


def image_show(img, pred):
    npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % pred)
    plt.show()


image_show(img_tensor.cpu(), pred + " " + str(score))

# Construct FGSM attacker
fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
perturbed_image_fgsm = fgsm.perturb(img_tensor, epsilon=0.16, target=285)
new_pred_fgsm, score_fgsm = get_prediction(model, perturbed_image_fgsm)

# inv_transform(img_tensor).shape


image_show(perturbed_image_fgsm.cpu(), new_pred_fgsm + " " + str(score_fgsm))

pgd = PGD(model, torch.nn.CrossEntropyLoss(reduction='none'), lower_bound=-1,
          upper_bound=1)  # construct the PGD attacker

perturbed_image_pgd = pgd.perturb(inputs=img_tensor, radius=0.13, step_size=0.02,
                                  step_num=7, target=torch.tensor([199]).to(device), targeted=True)
new_pred_pgd, score_pgd = get_prediction(model, perturbed_image_pgd)

image_show(perturbed_image_pgd.cpu(), new_pred_pgd + " " + str(score_pgd))

# Feature Ablation


feature_mask = torch.arange(64 * 7 * 7).reshape(8 * 7, 8 * 7).repeat_interleave(repeats=4, dim=1).repeat_interleave(
    repeats=4, dim=0).reshape(1, 1, 224, 224)
print(feature_mask)

from captum.attr import FeatureAblation

model.cpu()
ablator = FeatureAblation(model)
attr = ablator.attribute(img_tensor.cpu(), target=285, feature_mask=feature_mask)
# Choose single channel, all channels have same attribution scores
pixel_attr = attr[:, 0:1]


def pixel_dropout(image, dropout_pixels):
    keep_pixels = image[0][0].numel() - int(dropout_pixels)
    vals, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
    return (pixel_attr < vals.item()) * image


from captum.robust import MinParamPerturbation

min_pert_attr = MinParamPerturbation(forward_func=model, attack=pixel_dropout, arg_name="dropout_pixels", mode="linear",
                                     arg_min=0, arg_max=1024, arg_step=16,
                                     preproc_fn=None, apply_before_preproc=True)

pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(img_tensor.cpu(), target=285, perturbations_per_eval=10)
print("Minimum Pixels Dropped:", pixels_dropped)

new_pred_dropout, score_dropout = get_prediction(model, pixel_dropout_im)

image_show(pixel_dropout_im.cpu(), new_pred_dropout + " " + str(score_dropout))

# Grad CAM


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

targets = [ClassifierOutputTarget(281)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)

from pytorch_grad_cam import GradCAMPlusPlus

cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)










