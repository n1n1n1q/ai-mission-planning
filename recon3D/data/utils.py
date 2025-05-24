"""
Data management utilities for 3D reconstruction.
"""

import open3d as o3d
import numpy as np
import torch


def visualize_pcds(
    *pcds,
    window_name="Point cloud",
):
    """
    Visualize multiple point clouds in a single Open3D window.
    """
    o3d.visualization.draw_geometries(
        pcds, window_name=window_name, width=800, height=600
    )


def to_pointcloud(points, colors=None):
    """
    Convert a set of points to an Open3D point cloud.
    """

    pcd = o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    pcd.points = pts

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
