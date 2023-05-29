FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /home

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-m", "app"]