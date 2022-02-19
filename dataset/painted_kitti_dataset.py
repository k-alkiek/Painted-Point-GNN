from dataset.kitti_dataset import KittiDataset, Points
from collections import namedtuple, defaultdict
import numpy as np
from os.path import isfile, join
from os.path import isfile, join
PaintedPoints = namedtuple('Points', ['xyz', 'attr', 'scores'])

class PaintedKittiDataset(KittiDataset):
    
    def get_scores(self, frame_idx):
        point_file = join(self._point_dir, self._file_list[frame_idx])+'.npy'
        velo_data = np.load(point_file)
        scores = velo_data[:,4:]
        
        return scores
    
    def get_cam_points_in_image_with_rgb_and_scores(self, frame_idx, downsample_voxel_size=None, calib=None, xyz_range=None):
        """Get camera points that are visible in image and append image color to the points as attributes and class scores."""
        scores = self.get_scores(frame_idx)
        
        if calib is None:
            calib = self.get_calib(frame_idx)
        cam_points = self.get_cam_points(frame_idx, downsample_voxel_size,
            calib = calib, xyz_range=xyz_range)
        front_cam_points_idx = cam_points.xyz[:,2] > 0.1
        front_cam_points = Points(cam_points.xyz[front_cam_points_idx, :],
            cam_points.attr[front_cam_points_idx, :])
        
        scores = scores[front_cam_points_idx, :]
        
        image = self.get_image(frame_idx)
        height = image.shape[0]
        width = image.shape[1]
        img_points = self.cam_points_to_image(front_cam_points, calib)
        img_points_in_image_idx = np.logical_and.reduce(
            [img_points.xyz[:,0]>0, img_points.xyz[:,0]<width,
             img_points.xyz[:,1]>0, img_points.xyz[:,1]<height])
        
        
        cam_points_in_img = Points(
            xyz = front_cam_points.xyz[img_points_in_image_idx,:],
            attr = front_cam_points.attr[img_points_in_image_idx,:])
        
        scores = scores[img_points_in_image_idx,:]
        
        cam_points_in_img_with_rgb = self.rgb_to_cam_points(cam_points_in_img,
            image, calib)
        
        return PaintedPoints(xyz=cam_points_in_img_with_rgb.xyz, attr=cam_points_in_img_with_rgb.attr, scores=scores)
    
#     def get_cam_points_in_image_with_rgb_and_scores(self, frame_idx, downsample_voxel_size=None, calib=None, xyz_range=None):
#         cam_points = self.get_cam_points_in_image_with_rgb(frame_idx, downsample_voxel_size, calib, xyz_range)
#         scores = self.get_scores(frame_idx)
        
#         return PaintedPoints(cam_points.xyz, cam_points.attr, scores)

def downsample_by_random_voxel(points, voxel_size, add_rnd3d=False):
    """Downsample the points using base_voxel_size at different scales"""
    xmax, ymax, zmax = np.amax(points.xyz, axis=0)
    xmin, ymin, zmin = np.amin(points.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)

    if not add_rnd3d:
        xyz_idx = (points.xyz - xyz_offset) // voxel_size
    else:
        xyz_idx = (points.xyz - xyz_offset +
            voxel_size*np.random.random((1,3))) // voxel_size
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
    keys = xyz_idx[:, 0] + xyz_idx[:, 1]*dim_x + xyz_idx[:, 2]*dim_y*dim_x
    num_points = xyz_idx.shape[0]

    voxels_idx = {}
    for pidx in range(len(points.xyz)):
        key = keys[pidx]
        if key in voxels_idx:
            voxels_idx[key].append(pidx)
        else:
            voxels_idx[key] = [pidx]

    downsampled_xyz = []
    downsampled_attr = []
    for key in voxels_idx:
        center_idx = random.choice(voxels_idx[key])
        downsampled_xyz.append(points.xyz[center_idx])
        downsampled_attr.append(points.attr[center_idx])
        downsampled_score.append(points.scores[center_idx])

    return PaintedPoints(xyz=np.array(downsampled_xyz),
        attr=np.array(downsampled_attr),
        scores=np.array(downsampled_score))
