<div align="center">

# Assignment 4 - Deployment for Demos



<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/sushant097/TSAI-Assignment4-Deployment-for-Demos
cd TSAI-Assignment4-Deployment-for-Demos

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

### Assignment Related
**To Train the Cifar10 Torch Script**

`python src/train_script.py experiment=cifar`

It saves the `model.script.pt` model inside `logs/train/runs/*`

Using best hyperparmeters searched on Assignment 3
```bash
My Final Optuna sweeper parameter search output:

name: optuna
best_params:
  model.optimizer._target_: torch.optim.SGD
  model.optimizer.lr: 0.03584594526879088
  datamodule.batch_size: 128
best_value: 0.8082000017166138
```

**Run the Tensorboard**
`tensorboard --logdir=logs/train/runs --bind_all`

**To run the inference Cifar10 scripted model**

`python .\src\demo_scripted.py ckpt_path=D:\EMLO_V2\Assignment\TSAI-Assignment4-Deployment-for-Demos\logs\train\runs\2022-09-29_07-05-14\model.script.pt
`

Pack the inference scripted model inside the dockerize folder.
The non-compressed docker image size is 1.09 GB.

**Build the docker for only prediction**

`make build` or `docker build -t deploy dockerize/`

**RUN the docker container**
`docker run -t -p 8080:8080 deploy:latest`

