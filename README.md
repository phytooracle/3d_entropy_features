# 3D Topological Data Analysis (TDA)
This container extracts TDA features that describe the shape of pointclouds. A vector describing each dimension (x, y,z):
* Persistence entropy
* Number of points 
* Amplitude
  * Landscape
  * Bottleneck
  * Wasserstein 
  * Betti 
  * Silhouette
  * Heat
  * Persistence Image

## Flags/Arguments 

* -p, --pointclouds | Input point clouds either single or list
* -c, --cpu | Number of CPUs to use for multiprocessing. It is recommended to be 0.5*(total available CPUs)
* -o, --outdir | Output directory for CSV file containing feature values
* -f, --filename | Output filename for CSV file containing feature values

Values are extracted using the [Giotto-TDA](https://giotto-ai.github.io/gtda-docs/0.5.1/index.html) Python package.
