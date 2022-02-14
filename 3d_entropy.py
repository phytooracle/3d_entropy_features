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
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude
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
                        #nargs='+',
                        metavar='str',
                        type=str,
                        required=True)
                        #default='')

    #parser.add_argument('-c',
    #                    '--cpu',
    #                    help='Number of CPUs to use for multiprocessing.',
    #                    metavar='cpu',
    #                    type=int,
    #                    required=True)

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

    return parser.parse_args()


# --------------------------------------------------
def get_paths(directory):

    ortho_list = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            if 'final.ply' in name:
                ortho_list.append(os.path.join(root, name))

    if not ortho_list:

        raise Exception(f'ERROR: No compatible images found in {directory}.')

    print(f'Images to process: {len(ortho_list)}')

    return ortho_list


# --------------------------------------------------
def open_pcd(pcd_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f'{pcd_path} opened.')
    pcd.estimate_normals()
    print('Normals estimated.')
    pcd.normalize_normals()
    print('Normals normalized.')
    
    return pcd


# --------------------------------------------------
def downsample_pcd(pcd, voxel_size):

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
    #hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    #hull, _ = o3d.geometry.compute_point_cloud_convex_hull(pcd)#.get_volume()
    #hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    #print(hull_ls.get_volume())
    hull_volume = hull.get_volume()
    print(f'Volume: {hull_volume}')
    return hull_volume


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
def get_persistance_diagram(pcd_array):

    # Calculate persistance diagram
    diagram = VietorisRipsPersistence(metric='euclidean', homology_dimensions=[0, 1, 2]).fit_transform(pcd_array[None, :, :])

    return diagram


# --------------------------------------------------
def get_persistance_entropy_features(diagram):

    pe_features = PersistenceEntropy().fit_transform(diagram)

    return pe_features


# --------------------------------------------------
def get_number_points_features(diagram):

    np_features = NumberOfPoints().fit_transform(diagram)

    return np_features


# --------------------------------------------------
def get_amplitude_features(diagram, metric_val='landscape'):
    
    am_features = Amplitude(metric=metric_val).fit_transform(diagram)

    return am_features


# # --------------------------------------------------
# def separate_features(features):

#     zero = features[0][0]
#     one = features[0][1]
#     two = features[0][2]

#     return zero, one, two


# --------------------------------------------------
def get_min_max(pcd):
    
    max_x, max_y, max_z = pcd.get_max_bound()
    min_x, min_y, min_z = pcd.get_min_bound()

    return max_x, max_y, max_z, min_x, min_y, min_z


# --------------------------------------------------
def process_one_pointcloud(pcd_path, voxel_size):

    df = pd.DataFrame()

    try:
        plant_dict = {}
        
        # Open and downsample pointcloud
        pcd = open_pcd(pcd_path)
        point_count = len(pcd.points)
        print(f'{os.path.basename(pcd_path)} has {point_count} points.')
        down_pcd = downsample_pcd(pcd, voxel_size)
        print('Point cloud downsampled.')
        plant_name = os.path.splitext(os.path.basename(os.path.dirname(pcd_path)))[0]
        print(plant_name)
        max_x, max_y, max_z, min_x, min_y, min_z = get_min_max(pcd)
        print('Min max bounds calculated.')

        # Calculate plant and bounding box volumes
        hull_vol = calculate_convex_hull_volume(pcd)
        print('Hull volume calculated.')
        obb_vol = calculate_oriented_bb_volume(pcd)
        print('Oriented bounding box volume calculated.')
        abb_vol = calculate_axis_aligned_bb_volume(pcd)
        print('Axis aligned bounding box volume calculated.')

        # Calculate persistance diagrams and entropy features
        pcd_array = convert_point_cloud_to_array(down_pcd)
        diagram = get_persistance_diagram(pcd_array)
        print('Persistance diagram calculated.')

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
            'persistence_entropy_0': get_persistance_entropy_features(diagram)[0][0],
            'persistence_entropy_1': get_persistance_entropy_features(diagram)[0][1], 
            'persistence_entropy_2': get_persistance_entropy_features(diagram)[0][2], 
            'number_points_0': get_number_points_features(diagram)[0][0],
            'number_points_1': get_number_points_features(diagram)[0][1],
            'number_points_2': get_number_points_features(diagram)[0][2],
            'amplitude_landscape_0': get_amplitude_features(diagram)[0][0],
            'amplitude_landscape_1': get_amplitude_features(diagram)[0][1],
            'amplitude_landscape_2': get_amplitude_features(diagram)[0][2],
            'amplitude_bottleneck_0': get_amplitude_features(diagram, metric_val='bottleneck')[0][0],
            'amplitude_bottleneck_1': get_amplitude_features(diagram, metric_val='bottleneck')[0][1],
            'amplitude_bottleneck_2': get_amplitude_features(diagram, metric_val='bottleneck')[0][2],
            'amplitude_wasserstein_0': get_amplitude_features(diagram, metric_val='wasserstein')[0][0],
            'amplitude_wasserstein_1': get_amplitude_features(diagram, metric_val='wasserstein')[0][1],
            'amplitude_wasserstein_2': get_amplitude_features(diagram, metric_val='wasserstein')[0][2],
            'amplitude_betti_0': get_amplitude_features(diagram, metric_val='betti')[0][0],
            'amplitude_betti_1': get_amplitude_features(diagram, metric_val='betti')[0][1],
            'amplitude_betti_2': get_amplitude_features(diagram, metric_val='betti')[0][2],
            'amplitude_silhouette_0': get_amplitude_features(diagram, metric_val='silhouette')[0][0],
            'amplitude_silhouette_1': get_amplitude_features(diagram, metric_val='silhouette')[0][1],
            'amplitude_silhouette_2': get_amplitude_features(diagram, metric_val='silhouette')[0][2],
            'amplitude_heat_0': get_amplitude_features(diagram, metric_val='heat')[0][0],
            'amplitude_heat_1': get_amplitude_features(diagram, metric_val='heat')[0][1],
            'amplitude_heat_2': get_amplitude_features(diagram, metric_val='heat')[0][2],
            'amplitude_persistence_image_0': get_amplitude_features(diagram, metric_val='persistence_image')[0][0],
            'amplitude_persistence_image_1': get_amplitude_features(diagram, metric_val='persistence_image')[0][1],
            'amplitude_persistence_image_2': get_amplitude_features(diagram, metric_val='persistence_image')[0][2],
        }

        df = pd.DataFrame.from_dict(plant_dict, orient='index')
        df.index.name = 'plant_name'

    except:
        pass


    return df


# --------------------------------------------------
def main():
    """Exract entropy features here."""

    args = get_args()
    
    pointcloud_list = get_paths(args.pointclouds)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    major_df = pd.DataFrame()        
    for pointcloud in pointcloud_list:
        df = process_one_pointcloud(pointcloud, args.voxel_size)
        major_df = major_df.append(df)
    #with multiprocessing.Pool(args.cpu) as p:
    #    df = p.map(process_one_pointcloud, pointcloud_list)
    #    major_df = major_df.append(df)

    major_df.to_csv(os.path.join(args.outdir, ''.join([args.filename, '.csv'])))


# --------------------------------------------------
if __name__ == '__main__':
    main()
