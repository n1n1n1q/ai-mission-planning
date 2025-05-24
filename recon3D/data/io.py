"""
IO utilities for saving and loading reconstruction data
"""
import os
import pickle
import torch

def save_output_dict(output_dict, filepath):
    """
    Save the output dictionary from Fast3R inference to a file.
    
    Args:
        output_dict (dict): The output dictionary from Fast3R inference.
        filepath (str): Path where to save the output dictionary.
    """

    processed_dict = {
        'preds': [],
        'views': []
    }

    for pred in output_dict['preds']:
        pred_cpu = {}
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                pred_cpu[k] = v.cpu()
            else:
                pred_cpu[k] = v
        processed_dict['preds'].append(pred_cpu)

    for view in output_dict['views']:
        view_cpu = {}
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view_cpu[k] = v.cpu()
            else:
                view_cpu[k] = v
        processed_dict['views'].append(view_cpu)

    with open(filepath, 'wb') as f:
        pickle.dump(processed_dict, f)
    
    print(f"Output dictionary saved to {filepath}")

def load_output_dict(filepath, device=None):
    """
    Load the output dictionary from a file.
    
    Args:
        filepath (str): Path to the saved output dictionary.
        device (torch.device, optional): Device to load tensors to. If None, 
                                        tensors remain on CPU.
    
    Returns:
        dict: The loaded output dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        output_dict = pickle.load(f)
    
    if device is not None:
        for pred in output_dict['preds']:
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    pred[k] = v.to(device)

        for view in output_dict['views']:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device)
    
    print(f"Output dictionary loaded from {filepath}")
    return output_dict
