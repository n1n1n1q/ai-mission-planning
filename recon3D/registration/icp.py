"""
ICP algorithm for point cloud registration.
"""

import numpy as np
import open3d as o3d

def icp(source, target, threshold=0.02, max_iterations=50):
    """
    Perform ICP with Open3D.
    """
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), 
        max_iteration=max_iterations)
    T = reg_icp.transformation
    return T