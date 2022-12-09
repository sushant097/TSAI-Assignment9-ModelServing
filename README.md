<div align="center">

# Assignment  9- Model Serving



<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Experiment that shows how to serve the model using torch serve.

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
`python src/train_script.py experiment=example_timm trainer=gpu datamodule.batch_size=10000 trainer=gpu`

**Run the docker**
`docker run -it --rm --net=host -v `pwd`:/opt/src pytorch/torchserve:latest bash
`
**Convert scripted model to mar file**

torch-model-archiver --model-name cifar_basic --version 1.0 --serialized-file ./model-store/model.script.pt --handler ./src/torch_handlers/cifar_handler.py --extra-files ./src/torch_handlers/cifar_classes/index_to_name.json

`docker run -it --rm --net=host -v `pwd`:/opt/src pytorch/torchserve:latest bash`
`cd /opt/src`
    
`torchserve --start --model-store model-store --models cifar=cifar_basic.mar`


`curl http://127.0.0.1:8080/predictions/cifar -T test_serve/image/1000_truck.png`
Output:
```
{
  "truck": 0.9999533891677856,
  "cat": 1.9977218471467495e-05,
  "automobile": 7.188617018982768e-06,
  "frog": 6.326001312118024e-06,
  "bird": 4.050424649904016e-06
}
```
```bash
git clone https://github.com/pytorch/serve
cd serve
pip install -U grpcio protobuf grpcio-tools

python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto

python ts_scripts/torchserve_grpc_client.py infer cifar ../test_serve/image/1000_truck.png

```

`torchserve --start --model-store model-store --models cifar=cifar_basic.mar --ts-config config.properties
`
tensorboard --logdir pytorch_profiler/cifar_basic/ --bind_all

