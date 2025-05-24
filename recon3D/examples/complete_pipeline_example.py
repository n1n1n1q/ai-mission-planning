"""
Complete example script for video processing with 3D reconstruction and unique object detection.
This script demonstrates how to process videos to create point clouds with unique objects.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from recon3D.pipeline.object_reconstruction_pipeline import process_video_with_objects
from recon3D.data.utils import visualize_pcds


def main():
    """Main function demonstrating the complete pipeline."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Configuration
    video_file = "assets/hackaton videos /IMG_2265.MOV"
    output_directory = "data/processed_complete_example"

    # Check if video file exists
    if not Path(video_file).exists():
        logger.error(f"Video file not found: {video_file}")
        logger.info("Please make sure the video file exists or update the path")
        return

    logger.info("=== Starting Complete Video Processing Pipeline ===")

    try:
        # Process video with comprehensive object detection
        # This will:
        # 1. Extract frames from video
        # 2. Detect and track objects across frames
        # 3. Perform 3D reconstruction
        # 4. Extract unique object point clouds
        # 5. Ensure each object appears only once in interest_clouds

        cloud_with_views = process_video_with_objects(
            video_path=video_file,
            output_dir=output_directory,
            confidence_threshold=0.25,  # Lower threshold to catch more objects
            reconstruction_confidence=65,  # Good balance for reconstruction quality
            target_classes=None,  # Detect all object classes
            frames_per_second=1,  # Extract 1 frame per second
        )

        logger.info("=== Pipeline Results ===")
        logger.info(
            f"Main point cloud contains: {len(cloud_with_views.pcd.points)} points"
        )
        logger.info(
            f"Number of unique objects found: {len(cloud_with_views.interest_clouds)}"
        )
        logger.info(f"Number of camera views: {len(cloud_with_views.views)}")
        logger.info(f"Number of camera poses: {len(cloud_with_views.poses)}")

        # Print details about each unique object
        for i, obj_cloud in enumerate(cloud_with_views.interest_clouds):
            points_count = len(obj_cloud.points)
            logger.info(f"Object {i+1}: {points_count} points")

        # Visualize the results
        logger.info("=== Visualization ===")
        logger.info("Displaying main point cloud and all unique object clouds")

        # Combine main cloud with object clouds for visualization
        all_clouds = [cloud_with_views.pcd] + cloud_with_views.interest_clouds
        visualize_pcds(*all_clouds)

        logger.info("=== Pipeline Completed Successfully ===")

        # Save final results summary
        results_summary_path = Path(output_directory) / "final_results.txt"
        with open(results_summary_path, "w") as f:
            f.write("=== Video Processing Results ===\n")
            f.write(f"Input video: {video_file}\n")
            f.write(f"Main point cloud points: {len(cloud_with_views.pcd.points)}\n")
            f.write(
                f"Unique objects detected: {len(cloud_with_views.interest_clouds)}\n"
            )
            f.write(f"Camera views: {len(cloud_with_views.views)}\n")
            f.write(f"Camera poses: {len(cloud_with_views.poses)}\n\n")

            f.write("Object Details:\n")
            for i, obj_cloud in enumerate(cloud_with_views.interest_clouds):
                f.write(f"  Object {i+1}: {len(obj_cloud.points)} points\n")

        logger.info(f"Results summary saved to: {results_summary_path}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.exception("Full error traceback:")
        return

    logger.info("Script completed successfully!")


def demo_with_person_detection():
    """
    Demo function focusing specifically on person detection.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    video_file = "assets/hackaton videos /IMG_2265.MOV"

    if not Path(video_file).exists():
        logger.error(f"Video file not found: {video_file}")
        return

    logger.info("=== Person Detection Demo ===")

    # Process video focusing only on people
    cloud_with_views = process_video_with_objects(
        video_path=video_file,
        output_dir="data/person_detection_demo",
        confidence_threshold=0.3,
        target_classes=["person"],  # Only detect people
        frames_per_second=0.5,  # Extract 1 frame every 2 seconds
    )

    logger.info(f"Found {len(cloud_with_views.interest_clouds)} unique people")

    # Visualize only the people point clouds
    if cloud_with_views.interest_clouds:
        visualize_pcds(*cloud_with_views.interest_clouds)
    else:
        logger.info("No people detected in the video")


def demo_with_multiple_classes():
    """
    Demo function detecting multiple specific object classes.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    video_file = "assets/hackaton videos /IMG_2265.MOV"

    if not Path(video_file).exists():
        logger.error(f"Video file not found: {video_file}")
        return

    logger.info("=== Multiple Classes Detection Demo ===")

    # Process video detecting people, cars, and bikes
    target_classes = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]

    cloud_with_views = process_video_with_objects(
        video_path=video_file,
        output_dir="data/multi_class_demo",
        confidence_threshold=0.25,
        target_classes=target_classes,
        frames_per_second=1,
    )

    logger.info(f"Found {len(cloud_with_views.interest_clouds)} unique objects")
    logger.info(f"Target classes were: {target_classes}")

    # Visualize results
    all_clouds = [cloud_with_views.pcd] + cloud_with_views.interest_clouds
    visualize_pcds(*all_clouds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Video processing with 3D reconstruction and object detection"
    )
    parser.add_argument(
        "--demo",
        choices=["full", "person", "multi"],
        default="full",
        help="Choose demo type: full (all objects), person (people only), multi (multiple classes)",
    )

    args = parser.parse_args()

    if args.demo == "full":
        main()
    elif args.demo == "person":
        demo_with_person_detection()
    elif args.demo == "multi":
        demo_with_multiple_classes()
