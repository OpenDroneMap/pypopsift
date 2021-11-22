# pypopsift
CUDA accelerated SIFT in Python

This library is a wrapper around [PopSift](https://github.com/alicevision/popsift) to compute SIFT keypoints and descriptors on the GPU using CUDA. It's written to be a drop-in replacement for existing OpenCV functions such as `cv2.FeatureDetector_create('SIFT')` and ` cv2.DescriptorExtractor_create('SIFT')`.

## Building

Requirements:
 * CUDA Toolkit >= 7
 * CMake >= 3.14
 * A C++11 capable compiler (g++ 5.4.0 works fine)

```
# git clone --recurse-submodules https://github.com/uav4geo/pypopsift
# cd pypopsift && mkdir build && cd build
# make -j8
```

To install the Python package:

```
# cd pypopsift
# pip install .
```

## Usage

```
import cv2
import numpy as np
from pypopsift import popsift

filename = "/path/to/image.JPG"
config = {
    'sift_peak_threshold': 0.1,
    'sift_edge_threshold': 10.0,
    'feature_min_frames': 8000,
    'feature_use_adaptive_suppression': False,
    'feature_process_size': 2048
}

def resized_image(image, config):
    """Resize image to feature_process_size."""
    max_size = config['feature_process_size']
    h, w, _ = image.shape
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return image

flags = cv2.IMREAD_COLOR
image = cv2.imread(filename, flags)

if image is None:
    raise IOError("Unable to load image {}".format(filename))

if len(image.shape) == 3:
    image[:, :, :3] = image[:, :, [2, 1, 0]]

assert len(image.shape) == 3
image = resized_image(image, config)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

points, desc = popsift(image.astype(np.uint8),  # values between 0, 1
                            peak_threshold=config['sift_peak_threshold'],
                            edge_threshold=config['sift_edge_threshold'],
                            target_num_features=config['feature_min_frames'])
print(points.shape)
print(points)
print(desc.shape)
print(desc)
```

## License

Mozilla Public License 2.0

## Acknowledgements

```
@inproceedings{Griwodz2018Popsift,
	 author = {Griwodz, Carsten and Calvet, Lilian and Halvorsen, P{\aa}l},
	 title = {Popsift: A Faithful SIFT Implementation for Real-time Applications},
	 booktitle = {Proceedings of the 9th {ACM} Multimedia Systems Conference},
	 series = {MMSys '18},
	 year = {2018},
	 isbn = {978-1-4503-5192-8},
	 location = {Amsterdam, Netherlands},
	 pages = {415--420},
	 numpages = {6},
	 doi = {10.1145/3204949.3208136},
	 acmid = {3208136},
	 publisher = {ACM},
	 address = {New York, NY, USA},
}
```
