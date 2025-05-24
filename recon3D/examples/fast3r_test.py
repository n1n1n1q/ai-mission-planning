"""
Fast3R test script
"""
import logging
from recon3D.reconstruction.model import *
from recon3D.data.utils import *

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Load data

    filepath = "data"
    images = load_data(filepath)
    logging.info(f"Loaded {len(images)} images from {filepath}")
    output_dict = inference(images)
    logging.info("Inference completed")
    point_clouds = extract_point_clouds(output_dict)
    logging.info(f"Extracted {len(point_clouds)} point clouds")
    # poses_c2w_batch, estimated_focals = extract_poses(output_dict)
    # logging.info("Camera poses and estimated focal lengths extracted")
    pcds = [to_pointcloud(pc) for pc in point_clouds]
    visualize_pcds(pcds[0])