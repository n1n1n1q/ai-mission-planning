"""
Reconstruction with Fast3R
"""
import os
import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference as fast3r_inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# --- Setup ---

model = Fast3R.from_pretrained("model")  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
model.eval()
lit_module.eval()

def load_data(filepath):
    """
    Load data from a file path.
    This function should be implemented to load your specific data format.
    """
    filelist = os.listdir(filepath)
    filelist = [os.path.join(filepath, f) for f in filelist if f.endswith('.jpg') or f.endswith('.png')]
    images = load_images(filelist, size=512, verbose=True)
    return images

def inference(images):
    """
    Perform inference on the images using the Fast3R model.
    """
    output_dict, profiling_info = fast3r_inference(
        images,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=True,
    )
    return output_dict

def extract_point_clouds(output_dict):
    """
    Extract point clouds from the output dictionary.
    """
    point_clouds = []
    for pred in output_dict['preds']:
        point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
        point_clouds.append(point_cloud)
    return point_clouds

def extract_poses(output_dict):
    """
    Extract camera positions from the output dictionary.
    """
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )
    camera_poses = poses_c2w_batch[0]
    return camera_poses