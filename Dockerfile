FROM python:3.9.7-slim-buster

WORKDIR src

COPY requirements.txt requirements.txt


RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html



Run pip3 install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY ./logs/train/runs/2022-09-29_07-05-14/model.script.pt model.script.pt

COPY src/* src/

EXPOSE 8080

ENTRYPOINT ["python", "src/demo_scripted.py"]