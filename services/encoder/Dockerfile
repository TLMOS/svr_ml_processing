FROM pokroy/python-cuda:3.10-12.1

RUN apt-get install git -y

WORKDIR /home

COPY ./requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install git+https://github.com/openai/CLIP.git

COPY . .

CMD ["python3", "-m", "app"]