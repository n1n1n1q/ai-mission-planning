"""
ICP algorithm for point cloud registration.
"""

from dataclasses import dataclass
import numpy as np
import open3d as o3d
from open3d import geometry

from recon3D.data.cloud import CloudWithViews


def icp(source, target, threshold=0.02, max_iterations=50):
    """
    Perform ICP with Open3D.
    """
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    reg_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        max_iteration=max_iterations,
    )
    T = reg_icp.transformation
    return T


def hausdorff_test(cloud1: CloudWithViews, cloud2: CloudWithViews) -> list[float]:
    if len(cloud1.interest_clouds) != len(cloud2.interest_clouds):
        raise ValueError(
            f"Number of interest clouds doesn't match: {len(cloud1.interest_clouds)} vs {len(cloud2.interest_clouds)}"
        )

    distances = []

    for i, (cloud1_interest, cloud2_interest) in enumerate(
        zip(cloud1.interest_clouds, cloud2.interest_clouds)
    ):
        cloud1_points = np.asarray(cloud1_interest.points)
        cloud2_points = np.asarray(cloud2_interest.points)

        cloud1_tree = o3d.geometry.KDTreeFlann(cloud1_interest)
        cloud2_tree = o3d.geometry.KDTreeFlann(cloud2_interest)

        distances_1_to_2 = []
        for point in cloud1_points:
            _, idx, dist = cloud2_tree.search_knn_vector_3d(point, 1)
            distances_1_to_2.append(np.sqrt(dist[0]))

        distances_2_to_1 = []
        for point in cloud2_points:
            _, idx, dist = cloud1_tree.search_knn_vector_3d(point, 1)
            distances_2_to_1.append(np.sqrt(dist[0]))

        directed_hausdorff_1_to_2 = np.max(distances_1_to_2) if distances_1_to_2 else 0
        directed_hausdorff_2_to_1 = np.max(distances_2_to_1) if distances_2_to_1 else 0
        hausdorff_distance = max(directed_hausdorff_1_to_2, directed_hausdorff_2_to_1)

        distances.append(hausdorff_distance)

    return distances


def compute_average_hausdorff(cloud1: CloudWithViews, cloud2: CloudWithViews) -> float:
    distances = hausdorff_test(cloud1, cloud2)
    return sum(distances) / len(distances) if distances else 0
