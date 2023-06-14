FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /home

COPY ./services/encoder/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./common ./common
COPY ./services/encoder .

CMD ["python3", "-m", "app"]