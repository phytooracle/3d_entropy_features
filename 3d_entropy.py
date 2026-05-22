#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2021-11-19
Purpose: 3D phenotype extraction
"""

import argparse
import os

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import ConvexHull


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='3D phenotype extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p',
                        '--pointclouds',
                        help='Input point clouds directory.',
                        metavar='pointcloud_dir',
                        type=str,
                        required=True)

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory for CSV file containing entropy values.',
                        metavar='outdir',
                        type=str,
                        default='3d_volumes_entropy')

    parser.add_argument('-f',
                        '--filename',
                        help='Output filename for CSV file containing entropy values.',
                        metavar='filename',
                        type=str,
                        default='3d_volumes_entropy')

    parser.add_argument('-v',
                        '--voxel_size',
                        help='Voxel size for point cloud downsampling.',
                        metavar='voxel_size',
                        type=float,
                        default=0.09)
    
    parser.add_argument('-t',
                        '--calculate_tda',
                        help='Whether to calculate the entropy values.',
                        action='store_true',
                        dest='calculate_tda')

    return parser.parse_args()


# --------------------------------------------------
def get_paths(directory):

    pcd_list = []

    for root, _, files in os.walk(directory):
        for name in files:
            if 'final.ply' in name:
                pcd_list.append(os.path.join(root, name))

    if not pcd_list:
        raise Exception(f'ERROR: No compatible point clouds found in {directory}.')

    print(f'Point clouds to process: {len(pcd_list)}')

    return pcd_list


# --------------------------------------------------
def open_pcd(pcd_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f'\t{pcd_path} opened.')
    pcd.estimate_normals()
    pcd.normalize_normals()
    
    return pcd


# --------------------------------------------------
def visualize_pcd(pcd, extra=None):

    if extra:
        o3d.visualization.draw_geometries([pcd, extra])
    else:    
        o3d.visualization.draw_geometries([pcd])


# --------------------------------------------------
def get_min_max(pcd):
    
    max_x, max_y, max_z = pcd.get_max_bound()
    min_x, min_y, min_z = pcd.get_min_bound()

    return max_x, max_y, max_z, min_x, min_y, min_z


# --------------------------------------------------
def process_one_pointcloud(pcd_path, calculate_tda, voxel_size):

    df = pd.DataFrame()

    try:
        plant_dict = {}
        
        plant_name = os.path.splitext(os.path.basename(os.path.dirname(pcd_path)))[0]
        print(f'Processing {plant_name}')
        
        # Open and downsample pointcloud
        pcd = open_pcd(pcd_path)
        point_count = len(pcd.points)
        print(f'\t{os.path.basename(pcd_path)} has {point_count} points.')
        max_x, max_y, max_z, min_x, min_y, min_z = get_min_max(pcd)

        # Shift the point cloud to have zero mean for all coordinates
        points = np.asarray(pcd.points)
        x_mean = np.mean(points[:,0])
        y_mean = np.mean(points[:,1])
        z_mean = np.mean(points[:,2])
        points[:,0] -= x_mean
        points[:,1] -= y_mean
        points[:,2] -= z_mean
        pcd.points = o3d.utility.Vector3dVector(points)

        # Calculate convex hull and bounding box volumes
        hull_vol = ConvexHull(points).volume
        obb_vol = pcd.get_oriented_bounding_box().volume()
        abb_vol = pcd.get_axis_aligned_bounding_box().volume()

        # Create dictionary of outputs
        plant_dict[plant_name] = {
            'num_points': point_count,
            'min_x': min_x,
            'min_y': min_y,
            'min_z': min_z,
            'max_x': max_x,
            'max_y': max_y,
            'max_z': max_z,
            'convex_hull_volume': hull_vol,
            'oriented_bounding_box_volume': obb_vol, 
            'axis_aligned_bounding_box_volume': abb_vol,
        }

        # TDA features
        if calculate_tda:
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            downsampled_point_count = len(downsampled_pcd.points)
            print(f'\tDownsampled point cloud has {downsampled_point_count} points.')
            
            dpcd_array = np.asarray(downsampled_pcd.points, dtype=float)
            diagram = VietorisRipsPersistence(metric='euclidean', homology_dimensions=[0, 1, 2]).fit_transform(dpcd_array[None, :, :])

            pe_features = PersistenceEntropy().fit_transform(diagram)
            np_features = NumberOfPoints().fit_transform(diagram)
            amp_landscape_features = Amplitude(metric='landscape').fit_transform(diagram)
            amp_bottleneck_features = Amplitude(metric='bottleneck').fit_transform(diagram)
            amp_wasserstein_features = Amplitude(metric='wasserstein').fit_transform(diagram)
            amp_betti_features = Amplitude(metric='betti').fit_transform(diagram)
            amp_silhouette_features = Amplitude(metric='silhouette').fit_transform(diagram)
            amp_heat_features = Amplitude(metric='heat').fit_transform(diagram)
            amp_persistence_image_features = Amplitude(metric='persistence_image').fit_transform(diagram)

            nested_dict = plant_dict[plant_name]
            nested_dict['num_downsampled_points'] = downsampled_point_count
            nested_dict['voxel_size'] = voxel_size
            nested_dict['persistence_entropy_0'] = pe_features[0][0]
            nested_dict['persistence_entropy_1'] = pe_features[0][1]
            nested_dict['persistence_entropy_2'] = pe_features[0][2]
            nested_dict['number_points_0'] = np_features[0][0]
            nested_dict['number_points_1'] = np_features[0][1]
            nested_dict['number_points_2'] = np_features[0][2]
            nested_dict['amplitude_landscape_0'] = amp_landscape_features[0][0]
            nested_dict['amplitude_landscape_1'] = amp_landscape_features[0][1]
            nested_dict['amplitude_landscape_2'] = amp_landscape_features[0][2]
            nested_dict['amplitude_bottleneck_0'] = amp_bottleneck_features[0][0]
            nested_dict['amplitude_bottleneck_1'] = amp_bottleneck_features[0][1]
            nested_dict['amplitude_bottleneck_2'] = amp_bottleneck_features[0][2]
            nested_dict['amplitude_wasserstein_0'] = amp_wasserstein_features[0][0]
            nested_dict['amplitude_wasserstein_1'] = amp_wasserstein_features[0][1]
            nested_dict['amplitude_wasserstein_2'] = amp_wasserstein_features[0][2]
            nested_dict['amplitude_betti_0'] = amp_betti_features[0][0]
            nested_dict['amplitude_betti_1'] = amp_betti_features[0][1]
            nested_dict['amplitude_betti_2'] = amp_betti_features[0][2]
            nested_dict['amplitude_silhouette_0'] = amp_silhouette_features[0][0]
            nested_dict['amplitude_silhouette_1'] = amp_silhouette_features[0][1]
            nested_dict['amplitude_silhouette_2'] = amp_silhouette_features[0][2]
            nested_dict['amplitude_heat_0'] = amp_heat_features[0][0]
            nested_dict['amplitude_heat_1'] = amp_heat_features[0][1]
            nested_dict['amplitude_heat_2'] = amp_heat_features[0][2]
            nested_dict['amplitude_persistence_image_0'] = amp_persistence_image_features[0][0]
            nested_dict['amplitude_persistence_image_1'] = amp_persistence_image_features[0][1]
            nested_dict['amplitude_persistence_image_2'] = amp_persistence_image_features[0][2]

        df = pd.DataFrame.from_dict(plant_dict, orient='index')
        df.index.name = 'plant_name'

    except Exception as e:
        print(f'Expection occurred while processing {plant_name} :: {e}')
        pass

    return df


# --------------------------------------------------
def main():
    """Extract entropy features here."""

    args = get_args()
    
    pointcloud_list = get_paths(args.pointclouds)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    major_df = pd.DataFrame()        

    N = len(pointcloud_list)
    for i, pointcloud in enumerate(pointcloud_list):
        df = process_one_pointcloud(pointcloud, args.calculate_tda, args.voxel_size)
        major_df = pd.concat([major_df, df])
        print(f'\t{i/N*100:.2f}% complete.')
    
    major_df.to_csv(os.path.join(args.outdir, ''.join([args.filename, '.csv'])))


# --------------------------------------------------
if __name__ == '__main__':
    main()
