FROM python:3.9.7-slim

WORKDIR src

COPY requirements.txt requirements.txt

#RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

Run pip3 install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

EXPOSE 8080

ENTRYPOINT ["python", "demo_traced.py"]