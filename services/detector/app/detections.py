from typing import Optional

from pydantic import BaseModel
import numpy as np

from common.config import settings
from common.database import redis
from app.utils import compute_iou, compute_ioa


class Detection(BaseModel):
    """
    Object detected in a frame.

    Attributes:
    - xyxy (list[int]): xyxy coordinates of the bounding box, size (4,)
    - score (float): confidence score of the detection
    - label (int): label of the detection
    - ftl (int): frame to live, number of frames to consider the detection
                 actual. This is used to fight 'flickering' detections.
    """
    box: list[int]
    score: float
    label: int
    ftl: int = settings.detector.detection_ftl


def get_detections(source_manager_id: str,
                   source_id: str) -> Optional[list[Detection]]:
    """
    Get actual detections for a source from Redis.
    Actual detections are detections from previous frames whuch ftl and
    ttl are not expired.
    - ftl - frames-to-live, fight 'flickering' detections
    - ttl - time-to-live, remove detections from inactive sources

    Parameters:
    - source_manager_id (str): id of the source manager
    - source_id (str): id of the source

    Returns:
    - detections (list[Detection]): list of detections for the source,
        or None if redis record does not exist
    """
    key = f'detector:detections:{source_manager_id}:{source_id}'
    detections = redis.json().get(key, '$')
    if detections:
        return [Detection(**d) for d in detections[0]]


def set_detections(source_manager_id: str, source_id: str,
                   detections: list[Detection]):
    """
    Set detections for a source in Redis.

    Parameters:
    - source_manager_id (str): id of the source manager
    - source_id (str): id of the source
    - detections (list[Detection]): list of detections for the source
    """
    detections = [d.dict() for d in detections]
    key = f'detector:detections:{source_manager_id}:{source_id}'
    redis.json().set(key, '$', detections)
    redis.expire(key, settings.detector.detection_ttl)


def get_new_detections(prev_detections: list[Detection],
                       detections: list[Detection]):
    """
    Filter out detections that were already present in the previous frames.
    Detections are considered the same if:
    a. Their IoU is greater than `iou_identity_threshold_hard
    b. They have the same label and their IoU is greater
       than `iou_identity_threshold_soft`

    Parameters:
    - prev_detections (list[Detection]): list of detections from the previous
        frames
    - detections (list[Detection]): list of detections from the current frame

    Returns:
    - new_detections (list[Detection]): list of detections from the current
        frame that are not in the previous frames
    - old_detections (list[Detection]): list of detections from the previous
        frames that are still in the current frame
    """
    if not prev_detections:
        return detections, []

    new_detections = []
    old_detections = []
    th_soft = settings.detector.iou_identity_threshold_soft
    th_hard = settings.detector.iou_identity_threshold_hard
    for detecion in detections:
        ious = compute_iou(np.array(detecion.box),
                           np.array([d.box for d in prev_detections]))
        max_iou = ious.max()
        best_match = prev_detections[ious.argmax()]
        if max_iou > th_hard or max_iou > th_soft and \
                detecion.label == best_match.label:
            old_detections.append(best_match)
        else:
            new_detections.append(detecion)
    return new_detections, old_detections


def is_child_detection(parent_detection: Detection,
                       child_detection: Detection) -> bool:
    """
    Check if a detection is a child of another detection.
    A detection is a child of another detection if:
    - it's class is in the class hierarchy of the parent detection
    - it's IoA is greater than `ioa_class_hierarchy_threshold`

    This can be used to create bigger bounding boxes for parent detections,
    so that they include their children. (e. g. a bag worn by a person
    can be included in the bounding box of the person)

    Parameters:
    - parent_detection (Detection): parent detection
    - child_detection (Detection): potential child detection
    - ioa (float): intersection over area of the child detection

    Returns:
    - is_child (bool): True if the child detection is a child of the parent
        detection, False otherwise
    """
    if parent_detection.label not in settings.detector.class_hierarchy:
        return False
    child_labels = settings.detector.class_hierarchy[parent_detection.label]
    if child_detection.label not in child_labels:
        return False
    ioa = compute_ioa(np.array(child_detection.box),
                      np.array(parent_detection.box).reshape(1, 4))[0]
    if ioa < settings.detector.ioa_class_hierarchy_threshold:
        return False
    return True
