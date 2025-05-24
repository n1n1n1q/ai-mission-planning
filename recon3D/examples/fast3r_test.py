"""
Fast3R test script
"""
import logging
from recon3D.reconstruction.model import *
from recon3D.data.utils import *
from recon3D.data.io import save_output_dict, load_output_dict


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    filepath = "data/photos"
    images = load_data(filepath)
    logging.info(f"Loaded {len(images)} images from {filepath}")
    output_dict = inference(images)
    print(output_dict.keys())
    print(output_dict["views"][0].keys())
    print(output_dict["preds"][0].keys())
    save_output_dict(output_dict, "saved_reconstruction.pkl")
    logging.info("Inference completed")
    estimate_global_poses(output_dict, confidence=20)
    pcd = merge_clouds(output_dict, confidence=20)
    visualize_pcds(pcd.pcd)

