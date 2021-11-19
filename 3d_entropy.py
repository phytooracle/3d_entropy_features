#!/usr/bin/env python3
"""
Author : eg
Date   : 2021-11-19
Purpose: Rock the Casbah
"""

import argparse
import os
import sys
import open3d as o3d
import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import multiprocessing
import glob


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p',
                        '--pointclouds',
                        help='Input point clouds either single or list.',
                        nargs='+',
                        metavar='str',
                        type=str,
                        default='')

    parser.add_argument('-c',
                        '--cpu',
                        help='Number of CPUs to use for multiprocessing.',
                        metavar='cpu',
                        type=int,
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

    return parser.parse_args()


# --------------------------------------------------
def open_pcd(pcd_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.estimate_normals()
    pcd.normalize_normals()
    
    return pcd


# --------------------------------------------------
def downsample_pcd(pcd, voxel_size=0.05):

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return down_pcd


# --------------------------------------------------
def visualize_pcd(pcd, extra=None):
    if extra:
        o3d.visualization.draw_geometries([pcd, extra])
    else:    
        o3d.visualization.draw_geometries([pcd])


# --------------------------------------------------
def calculate_convex_hull_volume(pcd):
    hull, _ = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_vol = hull.get_volume()

    return hull_vol


# --------------------------------------------------
def calculate_oriented_bb_volume(pcd):

    obb_vol = pcd.get_oriented_bounding_box().volume()

    return obb_vol


# --------------------------------------------------
def calculate_axis_aligned_bb_volume(pcd):

    abb_vol = pcd.get_axis_aligned_bounding_box().volume()

    return abb_vol


# --------------------------------------------------
def convert_point_cloud_to_array(pcd):
    # Convert point cloud to a Numpy array
    pcd_array = np.asarray(pcd.points)
    pcd_array  = pcd_array.astype(float)

    return pcd_array


# --------------------------------------------------
def calculate_persistance_diagram(pcd_array):
    # Calculate persistance diagram
    VR = VietorisRipsPersistence(metric='euclidean', homology_dimensions=[0, 1, 2])  # Parameter explained in the text
    diagrams = VR.fit_transform(pcd_array[None, :, :])

    # Calculate the entropy
    PE = PersistenceEntropy()
    features = PE.fit_transform(diagrams)

    return features


# --------------------------------------------------
def separate_features(features):

    zero = features[0][0]
    one = features[0][1]
    two = features[0][2]

    return zero, one, two


# --------------------------------------------------
def get_min_max(pcd):
    
    max_x, max_y, max_z = pcd.get_max_bound()
    min_x, min_y, min_z = pcd.get_min_bound()

    return max_x, max_y, max_z, min_x, min_y, min_z


# --------------------------------------------------
def process_one_pointcloud(pcd_path):

    plant_dict = {}
    
    # Open and downsample pointcloud
    pcd = open_pcd(pcd_path)
    point_count = len(pcd.points)
    print(f'{os.path.basename(pcd_path)} has {point_count} points.')
    down_pcd = downsample_pcd(pcd)
    plant_name = os.path.splitext(os.path.basename(os.path.dirname(pcd_path)))[0]

    max_x, max_y, max_z, min_x, min_y, min_z = get_min_max(pcd)

    # Calculate plant and bounding box volumes
    hull_vol = calculate_convex_hull_volume(pcd)
    obb_vol = calculate_oriented_bb_volume(pcd)
    abb_vol = calculate_axis_aligned_bb_volume(pcd)

    # Calculate persistance diagrams and entropy features
    pcd_array = convert_point_cloud_to_array(down_pcd)
    features = calculate_persistance_diagram(pcd_array)
    zero, one, two = separate_features(features)

    # Create dictionary of outputs
    plant_dict[plant_name] = {
        'min_x': min_x,
        'min_y': min_y,
        'min_z': min_z,
        'max_x': max_x,
        'max_y': max_y,
        'max_z': max_z,
        'num_points': point_count,
        'hull_volume': hull_vol,
        'oriented_bounding_box': obb_vol, 
        'axis_aligned_bounding_box': abb_vol, 
        'persistence entropies_feature_0': zero,
        'persistence entropies_feature_1': one, 
        'persistence entropies_feature_2': two  
    }

    df = pd.DataFrame.from_dict(plant_dict, orient='index')

    return df


# --------------------------------------------------
def main():
    """Exract entropy features here."""

    args = get_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    major_df = pd.DataFrame()        
    # with multiprocessing.Pool(args.cpu) as p:
    #     df = p.map(process_one_pointcloud, args.pointclouds)
    #     major_df = major_df.append(df)

    for pcd in args.pointclouds:
        df = process_one_pointcloud(pcd)
        major_df = major_df.append(df)

    major_df.to_csv(os.path.join(args.outdir, ''.join([args.filename, '.csv'])))


# --------------------------------------------------
if __name__ == '__main__':
    main()
