from io import BytesIO
from PIL import Image

import numpy as np


def compute_iou(box, boxes) -> np.ndarray:
    """
    Compute IoU between a bounding box and a list of bounding boxes.

    Parameters:
    - box (np.ndarray): bounding box (xyxy), shape (4,)
    - boxes (np.ndarray): bounding boxes, shape (N, 4)

    Returns:
    - iou (np.ndarray): IoU, shape (N,)
    """
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area

    return iou


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding box format from (xywh) to (xyxy)

    Parameters:
    - box (np.ndarray): bounding box in (xywh) format,
        shape (N, 4) or (4,) for single box

    Returns:
    - transformed (np.ndarray): bounding box in (xyxy) format
    """
    transformed = np.copy(boxes)
    transformed[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    transformed[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    transformed[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    transformed[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return transformed


def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert image to bytes.

    Parameters:
    - image (np.ndarray): image to convert

    Returns:
    - bytes: image as bytes
    """
    image = Image.fromarray(image)
    with BytesIO() as buffer:
        image.save(buffer, format='JPEG')
        return buffer.getvalue()


def crop_image(image: np.ndarray, box: list) -> np.ndarray:
    """
    Crop image by box.

    Parameters:
    - image (np.ndarray): image to crop
    - box (list): xyxy coordinates of box

    Returns:
    - cropped_image (np.ndarray): cropped image
    """
    box = list(map(int, box))
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(image.shape[1], box[2])
    box[3] = min(image.shape[0], box[3])
    return image[box[1]:box[3], box[0]:box[2]]
