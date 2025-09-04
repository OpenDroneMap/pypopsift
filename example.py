import cv2
import numpy as np
from pypopsift import popsift
import time

print("Loading image...")

files = [
    "/datasets/brighton2/images/DJI_0018.JPG",
    "/datasets/brighton2/images/DJI_0019.JPG",
]
config = {
    'sift_peak_threshold': 0.1,
    'sift_edge_threshold': 10.0,
    'feature_min_frames': 24000,
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

for filename in files:
    flags = cv2.IMREAD_COLOR
    image = cv2.imread(filename, flags)

    if image is None:
        raise IOError("Unable to load image {}".format(filename))

    if len(image.shape) == 3:
        image[:, :, :3] = image[:, :, [2, 1, 0]]

    assert len(image.shape) == 3
    image = resized_image(image, config)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    print("Computing features...")
    start = time.time()
    points, desc = popsift(image,
                            peak_threshold=config['sift_peak_threshold'],
                            edge_threshold=config['sift_edge_threshold'],
                            target_num_features=config['feature_min_frames'],
                            downsampling=-1)

    print(points.shape)
    print(points)
    print(desc.shape)
    print(desc)
    print("Features computed in {:.3f} seconds".format(time.time() - start))
