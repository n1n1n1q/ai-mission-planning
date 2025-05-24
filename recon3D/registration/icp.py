"""
ICP algorithm for point cloud registration.
"""

import numpy as np
import open3d as o3d

from recon3D.data.cloud import CloudWithViews
from open3d import geometry


def icp(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    threshold=0.02,
    voxel_size=0.02,
    max_iterations=50,
):
    """
    Perform ICP with Open3D.
    """
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)

    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100),
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100),
    )

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        src_fpfh,
        tgt_fpfh,
        True,  # mutual_filter parameter
        max_correspondence_distance=1.5 * voxel_size,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                1.5 * voxel_size
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=4_000, confidence=0.999
        ),
    )

    T_init = ransac.transformation

    reg_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,  # max_correspondence_distance
        T_init,  # init transformation
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    T = reg_icp.transformation
    return T
