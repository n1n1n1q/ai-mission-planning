"""
Slam utilities
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# =========================================
# INTRINSIC CAMERA PARAMETERS
# =========================================
CAM_FX = 800.0
CAM_FY = 800.0
CAM_CX = 640.0
CAM_CY = 360.0

K = np.array([[CAM_FX,     0.0, CAM_CX],
              [0.0,     CAM_FY, CAM_CY],
              [0.0,        0.0,    1.0]])

# =========================================

@dataclass
class DetectedObject:
    """Class for detected objects with bounding box and class information."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    associated_keypoints: List[int] = None
    
    def __post_init__(self):
        if self.associated_keypoints is None:
            self.associated_keypoints = []

@dataclass
class KeypointInfo:
    """Class for storing keypoint information."""
    keypoint: cv2.KeyPoint
    descriptor: np.ndarray
    is_in_object: bool = False
    object_id: int = -1
    object_class: str = ""

def keypoint_extractor(image):
    """
    Extracts keypoints from the image using ORB feature detector.
    
    Args:
        image: Input image (grayscale)
        
    Returns:
        Tuple of (keypoints, descriptors)
    """

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        patchSize=31
    )
    
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    return keypoints, descriptors

def keypoints_matcher(descriptors1, descriptors2, ratio_thresh=0.75):
    """
    Matches keypoints between two images using FLANN-based matcher.
    
    Args:
        descriptors1: Descriptors from first image
        descriptors2: Descriptors from second image
        ratio_thresh: Ratio test threshold
        
    Returns:
        List of good matches
    """
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=1
    )
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    return good_matches

def is_point_in_bbox(point, bbox):
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates
        bbox: (x_min, y_min, width, height)
        
    Returns:
        Boolean indicating if point is inside bbox
    """
    x, y = point
    x_min, y_min, width, height = bbox
    
    return (x_min <= x <= x_min + width) and (y_min <= y <= y_min + height)

def associate_keypoints_with_objects(keypoints, detected_objects):
    """
    Associate keypoints with detected objects if they are inside object bounding boxes.
    
    Args:
        keypoints: List of keypoints
        detected_objects: List of DetectedObject instances
        
    Returns:
        List of KeypointInfo with object associations
    """
    keypoint_info_list = []
    
    for i, keypoint in enumerate(keypoints):
        kp_info = KeypointInfo(keypoint=keypoint, descriptor=None)

        for obj_id, obj in enumerate(detected_objects):
            if is_point_in_bbox((int(keypoint.pt[0]), int(keypoint.pt[1])), obj.bbox):
                kp_info.is_in_object = True
                kp_info.object_id = obj_id
                kp_info.object_class = obj.class_name
                obj.associated_keypoints.append(i)
                break
        
        keypoint_info_list.append(kp_info)
    
    return keypoint_info_list
