import waterz
import daisy
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
from skimage.morphology import watershed

def find_seeds(dist, sigma=3.0):
    smoothed = gaussian_filter(dist, sigma=sigma)
    return label(peak_local_max(smoothed, indices=False))[0]

def normalize_distances(predicted_distances):
    normalized_distances = predicted_distances.astype(np.float32)/255.0
    normalized_distances -= 126/255.0
    normalized_distances[normalized_distances<0] = 0
    return normalized_distances

def blockwise_segmentation_function(array_in, roi, thresholds):
	#Materialize region of interest
	predicted_distances = array_in.to_ndarray(roi, fill_value=0)

	#Normalizing distances
	normalized_distances = normalize_distances(predicted_distances)
	predicted_distances = None

	#Find seeds
	seeds = find_seeds(normalized_distances)

	#Watershed fragments
	fragments = watershed(-normalized_distances, seeds, mask = (normalized_distances.astype(bool))).astype(np.uint64)

	#Creating affinity arrays
	affs = np.stack([normalized_distances.data]*3)

	#Agglolmerate
	agglomeration = []
	for s in waterz.agglomerate(
	    affs=affs,
	    fragments=fragments,
	    thresholds=[thresholds],
	    scoring_function = 1 - waterz.QuantileAffinity(50)):
		agglomeration = s

	#fragments = np.ones((182,182,182),np.uint64)
	return agglomeration