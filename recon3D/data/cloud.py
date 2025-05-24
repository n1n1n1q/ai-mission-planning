"""
Point cloud data structure
"""

from dataclasses import dataclass
from open3d import geometry

@dataclass
class CloudWithViews:
    pcd: geometry.PointCloud
    views: list
    poses: list
    interesting_clouds: dict = None
    obj_frames: dict = None
    def __matmul__(self, B):
        """
        Multiply PCD's by a 4x4 matrix
        """
        self.pcd.transform(B)
        for k in range(len(self.views)):
            self.poses[k] = B @ self.poses[k]
        return self