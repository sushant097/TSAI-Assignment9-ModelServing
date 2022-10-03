FROM python:3.9.7-slim-buster

WORKDIR /app/

COPY requirements.txt requirements.txt

RUN pip3 install Cython

RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


# RUN pip3 install hydra-core --upgrade

RUN pip3 install -r requirements.txt

COPY "D:\\EMLO_V2\Assignment\TSAI-Assignment4-Deployment-for-Demos\logs\train\runs\2022-09-29_07-05-14\model.script.pt" "model.script.pt"

COPY src/* .

ENTRYPOINT ["python", ".\src\demo_scripted.py"]