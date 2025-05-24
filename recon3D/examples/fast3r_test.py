"""
Fast3R test script
"""
import logging
from recon3D.reconstruction.model import *
from recon3D.data.utils import *

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    filepath = "data"
    images = load_data(filepath)
    logging.info(f"Loaded {len(images)} images from {filepath}")
    output_dict = inference(images)
    logging.info("Inference completed")
    pcd = merge_clouds(output_dict)
    visualize_pcds(pcd)
