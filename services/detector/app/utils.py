import numpy as np


def intersection_over_union(box_a: list, box_b: list) -> float:
    """
    Calculate intersection over union of two boxes.

    Parameters:
    - box_a (list): xyxy coordinates of box A
    - box_b (list): xyxy coordinates of box B

    Returns:
    - iou (float): intersection over union of box A and box B
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


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
    return image[box[1]:box[3], box[0]:box[2]]
