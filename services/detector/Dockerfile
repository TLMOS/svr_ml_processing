FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /home

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./services/detector/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./common ./common
COPY ./services/detector .

CMD ["python3", "-m", "app"]