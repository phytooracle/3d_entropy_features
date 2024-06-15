# 3D Topological Data Analysis (TDA)
This container extracts TDA features using the [Giotto-TDA](https://giotto-ai.github.io/gtda-docs/0.5.1/index.html) Python package. A vector describing each dimension (x, y, z) is output:
* Persistence entropy [0, 1, 2]
* Number of points [0, 1, 2]
* Amplitude
  * Landscape [0, 1, 2]
  * Bottleneck [0, 1, 2]
  * Wasserstein [0, 1, 2]
  * Betti [0, 1, 2]
  * Silhouette [0, 1, 2]
  * Heat [0, 1, 2]
  * Persistence Image [0, 1, 2]

The units of traits are as follow:
  * Min_x, min_y, min_z, max_x, max_y, and max_z are UTM coordinates, so they are in units of meters
  * Num_points is the total count of points in the point cloud
  * Hull volume, oriented bounding box, and axis aligned bounding box are in units of cubic meters
  * The rest are topological values (those listed above) and are unitless.

## Flags/Arguments 

* -p, --pointclouds | Input point clouds either single or list
* -c, --cpu | Number of CPUs to use for multiprocessing. It is recommended to be 0.5*(total available CPUs)
* -o, --outdir | Output directory for CSV file containing feature values
* -f, --filename | Output filename for CSV file containing feature values
