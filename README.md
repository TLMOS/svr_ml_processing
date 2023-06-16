# SVR ML Processing
Before reading further, make sure you are familiar with the main [SVR repository](https://github.com/TLMOS/security_video_retrieval).

## Overview
SVR ML Processing is a set of services that perform video processing for SVR system. It consists of the following services:
- **Detector** - performs object detection on video chunks and returns frame crops.
- **Encoder** - encodes frame crops into image embeddings.

### Detector
Detector is a service that performs object detection on video chunks. It uses YOLOv8 model for object detection. It receives video chunks from RabbitMQ and returns frame crops to RabbitMQ.

Detector has a few features that make it more suitable for SVR system:
- **Parent-Child relationship between detections** - Parent detection bounding box is expanded to include all child detections. It is used to capture more context around the object. For example, if there is a person with a bag, the bounding box of the person will be expanded to include the bag. Two detections are considered to be a parent and a child if both of the following conditions are met:
    - IoA (Intersection over Area) of the child detection and the parent detection is greater than given threshold (0.3 by default)
    - Child detection class is in a list of possible child classes for the parent detection class (e.g. bag is a possible child class for person, not vice versa)
- **New detections only** - Detections that were already present on the previous frame are not sent to further processing. It is used to reduce load on the system. For example, if a car is parked in the same place for a long time, it would not be sent to further processing on every frame. Previous detections are stored in Redis. Detection is considered to be old if at least one of the following conditions is met:
    - There is a detection among stored previous detections that has the same class and IoU (Intersection over Union) with the current detection greater than soft threshold (0.45 by default)
    - There is a detection among stored previous detections which IoU with the current detection is greater than hard threshold (0.85 by default).
- **Detection FTL** - Detection FTL (Frames To Live) is a number of frames for which detection is stored in the system. It is used to improve `New detections only` feature by fighting "flickering detections" problem.

### Encoder
Encoder is a service that encodes frame crops into image embeddings. It uses CLIP model for image embedding. It receives frame crops from RabbitMQ and returns image embeddings to RabbitMQ.

## Deployment
Essential ML Processing configuration is stored in `.env` file. You can find full list of configuration variables in `common/config.py` file.

Before deploying ML Processing, you need to deploy RabbitMQ and Redis. You can find deployment instructions in their respective README files.

After that, specify RabbitMQ and Redis credentials in `.env` file. Default usernames and passwords are already set in `.env` file, so you can use them if you didn't change them during RabbitMQ and Redis deployment. But you still need to specify RabbitMQ and Redis hostnames. (And possibly ports, if you changed them during deployment.)

After that, you can deploy Search Engine using docker-compose:
```bash
docker-compose up --build -d
```

By default, ML Processing services are deployed with 1 replica each and with an access to GPU. If you want to change that, you can do it in `docker-compose.yml` file.