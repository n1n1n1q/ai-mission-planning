"""
Visual SLAM implementation
"""

import cv2
import numpy as np
from collections import defaultdict
import time
import logging
import os
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO

from .utils import (
    keypoint_extractor, 
    keypoints_matcher, 
    K, 
    DetectedObject, 
    KeypointInfo,
    is_point_in_bbox,
    associate_keypoints_with_objects
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualSLAM:
    """
    Visual SLAM implementation with object detection integration.
    """
    def __init__(self, camera_matrix: np.ndarray = None):
        """
        Initialize the Visual SLAM system.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_matrix = camera_matrix if camera_matrix is not None else K

        self.model = YOLO()
        self.confidence_threshold = 0.5

        self.prev_frame = None
        self.prev_kps = None
        self.prev_des = None
        self.prev_objects = []

        self.keyframes = []
        self.map_points = []
        self.object_map = defaultdict(list)

        self.current_pose = np.eye(4)
        self.pose_history = [self.current_pose.copy()]
        
        logger.info("Visual SLAM system initialized")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectedObject]]:
        """
        Process a new frame for SLAM and object detection.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed frame with visualizations, detected objects)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.model(frame, conf=self.confidence_threshold)
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                class_id = int(box.cls)
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                # if confidence > self.confidence_threshold and class_name in self.allowed_classes:
                detected_objects.append(
                    DetectedObject(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x, y, w, h)
                    )
                )
        keypoints, descriptors = keypoint_extractor(gray)
        keypoint_info_list = associate_keypoints_with_objects(keypoints, detected_objects)
        vis_frame = frame.copy()

        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kps = keypoints
            self.prev_des = descriptors
            self.prev_objects = detected_objects

            self._draw_features(vis_frame, keypoints, detected_objects)
            return vis_frame, detected_objects

        if self.prev_des is not None and descriptors is not None and len(self.prev_des) > 0 and len(descriptors) > 0:
            matches = keypoints_matcher(self.prev_des, descriptors)
            
            if len(matches) > 8:
                prev_pts = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
                curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])

                E, mask = cv2.findEssentialMat(
                    prev_pts, curr_pts, self.camera_matrix, 
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )

                if E is not None and E.shape == (3, 3):
                    _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts, self.camera_matrix, mask=mask)

                    transform = np.eye(4)
                    transform[:3, :3] = R
                    transform[:3, 3] = t.reshape(3)

                    self.current_pose = self.current_pose @ transform
                    self.pose_history.append(self.current_pose.copy())

                    if self._is_keyframe(matches):
                        self._add_keyframe(gray, keypoints, descriptors, detected_objects)
                        self._triangulate_new_points()
                        self._update_object_map(detected_objects)

        self._draw_features(vis_frame, keypoints, detected_objects)
        self._draw_trajectory(vis_frame)

        self.prev_frame = gray
        self.prev_kps = keypoints
        self.prev_des = descriptors
        self.prev_objects = detected_objects
        
        return vis_frame, detected_objects
    
    def _is_keyframe(self, matches) -> bool:
        """
        Determine if current frame should be a keyframe.
        """
        return len(self.keyframes) == 0 or len(matches) > 0.75 * \
            len(self.prev_kps) or len(self.keyframes) % 5 == 0
    
    def _add_keyframe(self, frame, keypoints, descriptors, objects):
        """
        Add a new keyframe to the map.
        """
        self.keyframes.append({
            'frame': frame,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': self.current_pose.copy(),
            'objects': objects
        })
        
        logger.info(f"Added keyframe #{len(self.keyframes)}")
    
    def _triangulate_new_points(self):
        """
        Triangulate 3D points from the most recent keyframes.
        """
        if len(self.keyframes) < 2:
            return

        kf1 = self.keyframes[-2]
        kf2 = self.keyframes[-1]
        

        if kf1['descriptors'] is not None and kf2['descriptors'] is not None:
            matches = keypoints_matcher(kf1['descriptors'], kf2['descriptors'])
            
            if len(matches) > 0:

                pts1 = np.float32([kf1['keypoints'][m.queryIdx].pt for m in matches])
                pts2 = np.float32([kf2['keypoints'][m.trainIdx].pt for m in matches])

                P1 = np.hstack((kf1['pose'][:3, :3], kf1['pose'][:3, 3:4]))
                P2 = np.hstack((kf2['pose'][:3, :3], kf2['pose'][:3, 3:4]))

                points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                
                points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]

                for i in range(points_3d.shape[1]):
                    self.map_points.append({
                        'point': points_3d[:, i],
                        'descriptor': kf1['descriptors'][matches[i].queryIdx],
                        'observed_in': [len(self.keyframes) - 2, len(self.keyframes) - 1]
                    })
    
    def _update_object_map(self, detected_objects):
        """Update the object map with new detections."""
        for obj in detected_objects:
            self.object_map[obj.class_name].append({
                'bbox': obj.bbox,
                'confidence': obj.confidence,
                'frame_idx': len(self.keyframes) - 1,
                'keypoints': obj.associated_keypoints
            })
    
    def _draw_features(self, frame, keypoints, objects):
        """Draw keypoints and objects on the frame."""
        for obj in objects:
            x, y, w, h = obj.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                f"{obj.class_name}: {obj.confidence:.2f}", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            color = (0, 0, 255)
            for obj in objects:
                if is_point_in_bbox((x, y), obj.bbox):
                    color = (255, 0, 0)
                    break
            
            cv2.circle(frame, (x, y), 3, color, -1)
    
    def _draw_trajectory(self, frame):
        """Draw the camera trajectory on the frame."""
        height, width = frame.shape[:2]
        scale = min(width, height) // 4
        offset_x = width - scale - 10
        offset_y = height - scale - 10

        cv2.line(frame, (offset_x, offset_y), (offset_x + scale//2, offset_y), (0, 0, 255), 2)  # X-axis
        cv2.line(frame, (offset_x, offset_y), (offset_x, offset_y - scale//2), (0, 255, 0), 2)  # Y-axis

        if len(self.pose_history) > 1:
            points = []
            for pose in self.pose_history:
                # Get camera position (inverse of pose)
                pos = -pose[:3, :3].T @ pose[:3, 3]
                # Scale and offset for visualization
                x = int(offset_x + pos[0] * scale)
                y = int(offset_y - pos[2] * scale)  # Using z for height
                points.append((x, y))
            
            # Draw path
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)
    
    def get_map_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current map."""
        return {
            'num_keyframes': len(self.keyframes),
            'num_map_points': len(self.map_points),
            'object_counts': {cls: len(objs) for cls, objs in self.object_map.items()},
            'camera_position': -self.current_pose[:3, :3].T @ self.current_pose[:3, 3]
        }