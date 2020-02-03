from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import daisy
import numpy as np

test_roi = daisy.Roi((3000, 0, 6000), (500, 500, 500))

test_file = '/nrs/saalfeld/heinrichl/cell/gt061719/unet/02-070219/hela_cell3_314000.n5'
mito_ds = 'mito'
er_ds = 'er'

def find_seeds(dist, sigma=3.0):

    smoothed = gaussian_filter(dist, sigma=sigma)
    return label(peak_local_max(smoothed, indices=False))[0]

def normalize_distances(array):

    array.data = array.data.astype(np.float32)/255.0
    array.data -= 0.5

if __name__ == '__main__':

    print("Opening datasets...")
    dist_er = daisy.open_ds(test_file, mito_ds)
    dist_mitos = daisy.open_ds(test_file, er_ds)

    print(f"Got distances in {dist_er.roi}")
    dist_er = dist_er[test_roi]
    dist_mitos = dist_mitos[test_roi]

    print("Reading into memory...")
    dist_er.materialize()
    dist_mitos.materialize()

    print("Normalizing distances...")
    normalize_distances(dist_er)
    normalize_distances(dist_mitos)

    # TODO: include all classes of the same level in the hierarchy
    print("Computing max distances...")
    dist_max = np.abs(np.maximum(dist_er.data, dist_mitos.data))

    print("Finding seeds...")
    seeds = find_seeds(dist_max)

    print("Running watershed...")
    print(f"dist_max: {dist_max.shape}")
    print(f"seeds: {seeds.shape}")
    fragments = watershed(-dist_max, seeds).astype(np.uint64)

    fragments = daisy.Array(
        data=fragments,
        roi=test_roi,
        voxel_size=dist_er.voxel_size)

    fragments_ds = daisy.prepare_ds(
        'test_er_mito.n5',
        'fragments',
        total_roi=fragments.roi,
        voxel_size=fragments.voxel_size,
        dtype=np.uint64)

    print("Storing fragments...")
    fragments_ds[test_roi] = fragments

    dist_max_ds = daisy.prepare_ds(
        'test_er_mito.n5',
        'dist_max',
        total_roi=fragments.roi,
        voxel_size=fragments.voxel_size,
        dtype=np.float32)

    print("Storing dist_max...")
    dist_max_ds[test_roi] = dist_max
