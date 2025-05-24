"""
Comprehensive pipeline for video processing with 3D reconstruction and object detection.
Ensures unique objects are stored only once in CloudWithViews.interest_clouds.
"""

import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import open3d as o3d
import torch

from recon3D.data.video_splitter import video_to_frames
from recon3D.reconstruction.model import load_data, inference, merge_clouds
from recon3D.object_detection.detector import Detector
from recon3D.data.cloud import CloudWithViews
from recon3D.data.io import save_output_dict, load_output_dict
from recon3D.data.utils import to_pointcloud


class VideoObjectTracker:
    """
    Tracks objects across video frames and manages unique object identification.
    """

    def __init__(self, confidence_threshold=0.3, iou_threshold=0.45, classes=None):
        self.detector = Detector(
            classes=classes, conf=confidence_threshold, iou=iou_threshold
        )
        self.tracked_objects = {}  # track_id -> object_info
        self.object_clouds = {}  # track_id -> point_cloud
        self.frame_detections = []  # All detections per frame

    def process_video_frames(self, frames_folder: str) -> Dict[int, List]:
        """
        Process all frames in a folder and track objects.
        Returns dictionary mapping frame_index to detections list.
        """
        frame_files = sorted(
            [
                f
                for f in os.listdir(frames_folder)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        all_detections = {}

        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            # Process frame with tracking
            detections = self.detector.process_frame_with_tracking(frame)
            all_detections[i] = detections

            # Update tracked objects
            for (
                track_id,
                x1,
                y1,
                x2,
                y2,
                conf,
                detected_class,
                class_name,
            ) in detections:
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        "first_frame": i,
                        "last_frame": i,
                        "class_name": class_name,
                        "class_id": detected_class,
                        "bbox_history": [(x1, y1, x2, y2, conf)],
                        "frame_indices": [i],
                    }
                else:
                    self.tracked_objects[track_id]["last_frame"] = i
                    self.tracked_objects[track_id]["bbox_history"].append(
                        (x1, y1, x2, y2, conf)
                    )
                    self.tracked_objects[track_id]["frame_indices"].append(i)

        self.frame_detections = all_detections
        return all_detections


class ReconstructionPipeline:
    """
    Main pipeline for video reconstruction with object detection and tracking.
    """

    def __init__(
        self,
        confidence_threshold=0.3,
        iou_threshold=0.45,
        reconstruction_confidence=65,
        target_classes=None,
    ):
        self.object_tracker = VideoObjectTracker(
            confidence_threshold, iou_threshold, target_classes
        )
        self.reconstruction_confidence = reconstruction_confidence
        self.logger = logging.getLogger(__name__)

    def extract_object_point_clouds(
        self,
        cloud_with_views: CloudWithViews,
        camera_intrinsics: Dict,
        frames_folder: str,
    ) -> CloudWithViews:
        """
        Extract point clouds for detected objects using depth projection.
        """
        frame_files = sorted(
            [
                f
                for f in os.listdir(frames_folder)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        unique_objects = {}  # track_id -> point_cloud

        # Get point cloud data
        points = np.asarray(cloud_with_views.pcd.points)
        colors = np.asarray(cloud_with_views.pcd.colors)

        # For each unique tracked object
        for track_id, obj_info in self.object_tracker.tracked_objects.items():
            object_points = []
            object_colors = []

            # Process frames where this object appears
            for frame_idx in obj_info["frame_indices"]:
                if frame_idx >= len(frame_files) or frame_idx >= len(
                    cloud_with_views.poses
                ):
                    continue

                # Get frame detection
                frame_detections = self.object_tracker.frame_detections.get(
                    frame_idx, []
                )
                object_detection = None

                for detection in frame_detections:
                    if detection[0] == track_id:  # track_id matches
                        object_detection = detection
                        break

                if object_detection is None:
                    continue

                track_id_det, x1, y1, x2, y2, conf, detected_class, class_name = (
                    object_detection
                )

                # Load frame image to get dimensions
                frame_path = os.path.join(frames_folder, frame_files[frame_idx])
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                height, width = frame.shape[:2]

                # Project 3D points to 2D and filter by bounding box
                pose = cloud_with_views.poses[frame_idx]

                # Simple projection (this should be refined based on actual camera parameters)
                # For now, we'll use a simplified approach
                frame_points, frame_colors = self._extract_points_in_bbox(
                    points, colors, pose, (x1, y1, x2, y2), (width, height)
                )

                if len(frame_points) > 0:
                    object_points.extend(frame_points)
                    object_colors.extend(frame_colors)

            # Create point cloud for this unique object
            if len(object_points) > 10:  # Minimum points threshold
                object_points = np.array(object_points)
                object_colors = np.array(object_colors)

                # Remove duplicates (points that are too close)
                object_points, object_colors = self._remove_duplicate_points(
                    object_points, object_colors, threshold=0.01
                )

                if len(object_points) > 5:
                    obj_pcd = to_pointcloud(object_points, object_colors)
                    unique_objects[track_id] = obj_pcd
                    self.logger.info(
                        f"Created point cloud for {obj_info['class_name']} (track_id: {track_id}) with {len(object_points)} points"
                    )

        # Add unique object clouds to interest_clouds
        cloud_with_views.interest_clouds = list(unique_objects.values())

        return cloud_with_views

    def _extract_points_in_bbox(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        pose: np.ndarray,
        bbox: Tuple,
        image_size: Tuple,
    ) -> Tuple[List, List]:
        """
        Extract 3D points that project within the 2D bounding box.
        This is a simplified implementation - should be improved with proper camera calibration.
        """
        x1, y1, x2, y2 = bbox
        width, height = image_size

        # Transform points to camera coordinate system
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        camera_points = (np.linalg.inv(pose) @ points_homogeneous.T).T[:, :3]

        # Simple perspective projection (assumes specific camera parameters)
        # This should be replaced with actual camera intrinsics
        focal_length = min(width, height)  # Simplified assumption
        cx, cy = width / 2, height / 2

        # Project to image plane
        with np.errstate(divide="ignore", invalid="ignore"):
            projected_x = (
                camera_points[:, 0] / camera_points[:, 2]
            ) * focal_length + cx
            projected_y = (
                camera_points[:, 1] / camera_points[:, 2]
            ) * focal_length + cy

        # Filter points within bounding box and in front of camera
        valid_mask = (
            (camera_points[:, 2] > 0)  # In front of camera
            & (projected_x >= x1)
            & (projected_x <= x2)
            & (projected_y >= y1)
            & (projected_y <= y2)
            & np.isfinite(projected_x)
            & np.isfinite(projected_y)
        )

        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask]

        return filtered_points.tolist(), filtered_colors.tolist()

    def _remove_duplicate_points(
        self, points: np.ndarray, colors: np.ndarray, threshold: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove duplicate points that are within threshold distance.
        """
        if len(points) == 0:
            return points, colors

        # Use Open3D for efficient duplicate removal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Remove duplicates
        pcd = pcd.remove_duplicated_points()

        # Voxel downsampling to further reduce density
        pcd = pcd.voxel_down_sample(voxel_size=threshold)

        return np.asarray(pcd.points), np.asarray(pcd.colors)

    def process_video(
        self,
        video_path: str,
        output_dir: str = None,
        frames_per_second: int = 1,
        save_intermediate: bool = True,
    ) -> CloudWithViews:
        """
        Complete pipeline to process video and create CloudWithViews with unique objects.
        """
        video_name = Path(video_path).stem
        if output_dir is None:
            output_dir = f"data/processed_{video_name}"

        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Extract frames from video
        self.logger.info(f"Extracting frames from {video_path}")
        frames_folder = os.path.join(output_dir, "frames")
        time_intervals, num_frames = video_to_frames(
            video_path, frames_folder, frames_per_second=frames_per_second
        )
        self.logger.info(f"Extracted {num_frames} frames")

        # Step 2: Process frames for object detection and tracking
        self.logger.info("Processing frames for object detection and tracking")
        detections = self.object_tracker.process_video_frames(frames_folder)
        self.logger.info(
            f"Detected {len(self.object_tracker.tracked_objects)} unique objects"
        )

        # Step 3: Load frames for 3D reconstruction
        self.logger.info("Loading frames for 3D reconstruction")
        images = load_data(frames_folder)

        # Step 4: Perform 3D reconstruction
        self.logger.info("Performing 3D reconstruction")
        output_dict = inference(images)

        if save_intermediate:
            reconstruction_path = os.path.join(output_dir, "reconstruction.pkl")
            save_output_dict(output_dict, reconstruction_path)
            self.logger.info(f"Saved reconstruction to {reconstruction_path}")

        # Step 5: Merge point clouds
        self.logger.info("Merging point clouds")
        cloud_with_views = merge_clouds(
            output_dict, confidence=self.reconstruction_confidence
        )

        # Step 6: Extract object point clouds
        self.logger.info("Extracting object point clouds")
        # For now, use simplified camera intrinsics - this should be improved
        camera_intrinsics = {"fx": 512, "fy": 512, "cx": 256, "cy": 256}

        cloud_with_views = self.extract_object_point_clouds(
            cloud_with_views, camera_intrinsics, frames_folder
        )

        # Step 7: Save results
        if save_intermediate:
            # Save detection results
            detections_path = os.path.join(output_dir, "detections.txt")
            self._save_detections(detections_path)

            # Save object summary
            summary_path = os.path.join(output_dir, "object_summary.txt")
            self._save_object_summary(summary_path)

        self.logger.info(
            f"Pipeline completed. Found {len(cloud_with_views.interest_clouds)} unique objects"
        )
        return cloud_with_views

    def _save_detections(self, filepath: str):
        """Save detection results to file."""
        with open(filepath, "w") as f:
            f.write("frame_id,track_id,x1,y1,x2,y2,confidence,class_id,class_name\n")
            for (
                frame_id,
                frame_detections,
            ) in self.object_tracker.frame_detections.items():
                for detection in frame_detections:
                    track_id, x1, y1, x2, y2, conf, class_id, class_name = detection
                    f.write(
                        f"{frame_id},{track_id},{x1},{y1},{x2},{y2},{conf},{class_id},{class_name}\n"
                    )

    def _save_object_summary(self, filepath: str):
        """Save object summary to file."""
        with open(filepath, "w") as f:
            f.write(
                "track_id,class_name,first_frame,last_frame,total_frames,avg_confidence\n"
            )
            for track_id, obj_info in self.object_tracker.tracked_objects.items():
                confidences = [bbox[4] for bbox in obj_info["bbox_history"]]
                avg_conf = np.mean(confidences)
                total_frames = len(obj_info["frame_indices"])
                f.write(
                    f"{track_id},{obj_info['class_name']},{obj_info['first_frame']},"
                    f"{obj_info['last_frame']},{total_frames},{avg_conf:.3f}\n"
                )


def process_video_with_objects(
    video_path: str,
    output_dir: str = None,
    confidence_threshold: float = 0.3,
    reconstruction_confidence: int = 65,
    target_classes: List[str] = None,
    frames_per_second: int = 1,
) -> CloudWithViews:
    """
    Convenience function to process a video and extract unique objects.

    Args:
        video_path: Path to input video
        output_dir: Directory to save intermediate results
        confidence_threshold: Object detection confidence threshold
        reconstruction_confidence: 3D reconstruction confidence threshold
        target_classes: List of class names to detect (None for all classes)
        frames_per_second: Frame extraction rate

    Returns:
        CloudWithViews with unique objects in interest_clouds
    """
    logging.basicConfig(level=logging.INFO)

    pipeline = ReconstructionPipeline(
        confidence_threshold=confidence_threshold,
        reconstruction_confidence=reconstruction_confidence,
        target_classes=target_classes,
    )

    return pipeline.process_video(
        video_path=video_path,
        output_dir=output_dir,
        frames_per_second=frames_per_second,
    )


if __name__ == "__main__":
    # Example usage
    video_file = "assets/hackaton videos /IMG_2265.MOV"

    # Process video with focus on people detection
    cloud_with_views = process_video_with_objects(
        video_path=video_file,
        output_dir="data/processed_example",
        confidence_threshold=0.3,
        target_classes=None,  # Only detect people
        frames_per_second=1,
    )

    print(f"Total unique objects found: {len(cloud_with_views.interest_clouds)}")
    print(f"Main point cloud has {len(cloud_with_views.pcd.points)} points")

    # Visualize results
    from recon3D.data.utils import visualize_pcds

    all_pcds = [cloud_with_views.pcd] + cloud_with_views.interest_clouds
    visualize_pcds(*all_pcds)
