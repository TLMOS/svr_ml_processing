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


def compute_ioa(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoA between a bounding box and a list of bounding boxes.

    Parameters:
    - box (np.ndarray): bounding box (xyxy), shape (4,)
    - boxes (np.ndarray): bounding boxes, shape (N, 4)

    Returns:
    - ioa (np.ndarray): IoA, shape (N,)
    """
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])

    ioa = intersection_area / box_area

    return ioa


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


def scale_boxes(boxes: np.ndarray, scale: float,
                max_size: tuple[int, int]) -> np.ndarray:
    """
    Scale bounding boxes.

    Parameters:
    - boxes (np.ndarray): bounding boxes (xyxy), shape (N, 4)
    - scale (float): scale factor
    - max_size (tuple[int, int]): maximum size of the image

    Returns:
    - scaled (np.ndarray): scaled bounding boxes
    """
    scaled = np.copy(boxes)
    w_pad = (boxes[:, 2] - boxes[:, 0]) * (scale - 1) / 2
    h_pad = (boxes[:, 3] - boxes[:, 1]) * (scale - 1) / 2
    scaled[:, 0] = boxes[:, 0] - w_pad
    scaled[:, 1] = boxes[:, 1] - h_pad
    scaled[:, 2] = boxes[:, 2] + w_pad
    scaled[:, 3] = boxes[:, 3] + h_pad
    scaled[:, 0] = np.clip(scaled[:, 0], 0, max_size[0])
    scaled[:, 1] = np.clip(scaled[:, 1], 0, max_size[1])
    scaled[:, 2] = np.clip(scaled[:, 2], 0, max_size[0])
    scaled[:, 3] = np.clip(scaled[:, 3], 0, max_size[1])
    return scaled


def outer_box(boxes: np.ndarray) -> np.ndarray:
    """
    Compute bounding box that contains all given bounding boxes.

    Parameters:
    - boxes (np.ndarray): bounding boxes (xyxy), shape (N, 4)

    Returns:
    - outer (np.ndarray): outer bounding box, shape (4,)
    """
    xmin = np.min(boxes[:, 0])
    ymin = np.min(boxes[:, 1])
    xmax = np.max(boxes[:, 2])
    ymax = np.max(boxes[:, 3])
    return np.array([xmin, ymin, xmax, ymax])


def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert image to bytes.

    Parameters:
    - image (np.ndarray): image to convert

    Returns:
    - bytes: image as bytes
    """
    image = Image.fromarray(image.astype('uint8'))
    with BytesIO() as buffer:
        image.save(buffer, format='PNG')
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
