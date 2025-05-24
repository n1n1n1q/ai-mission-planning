"""
Point cloud data structure
"""

from dataclasses import dataclass
from open3d import geometry

@dataclass
class CloudWithViews:
    pcd: geometry.PointCloud
    views: dict
