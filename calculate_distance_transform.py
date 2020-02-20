from scipy.ndimage import gaussian_filter, label, distance_transform_edt
import scipy.misc
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import daisy
import numpy as np

test_roi = daisy.Roi((0, 0, 0), (501, 501, 501))

test_file = '/groups/cosem/cosem/ackermand/HeLa_Cell3_4x4x4nm_it450000_crop.n5'
mito_ds = 'mito'

if __name__ == '__main__':

    print("Opening datasets...")
    dist_mitos = daisy.open_ds(test_file, mito_ds)

    print(f"Got distances in {dist_mitos.roi}")
    dist_mitos = dist_mitos[test_roi]

    print("Reading into memory...")
    dist_mitos.materialize()

    print("Calculating distances in voxels")
    predicted_distance_in_voxels = np.arctanh( (dist_mitos.data - 127) / 128 ) * 50/4
 
    print("Binarize...")
    binarized = dist_mitos.data>=127

    print("Mask original data...")
    predicted_distance_transform_masked = dist_mitos.data
    predicted_distance_transform_masked[~binarized] = 0



    print("Calculate distance transform...")
    distance_transform = distance_transform_edt(binarized)
    actual_distance_transform_masked = 128*np.tanh(distance_transform*4/50) + 127
    actual_distance_transform_masked[~binarized] = 0
    
    binarized_daisy = daisy.Array(
        data=binarized,
        roi=test_roi,
        voxel_size=dist_mitos.voxel_size)

    binarized_ds = daisy.prepare_ds(
        'test_mito.n5',
        'binarized',
        total_roi=binarized_daisy.roi,
        voxel_size=binarized_daisy.voxel_size,
        dtype=np.uint64)

    print("Storing binarized...")
    binarized_ds[test_roi] = binarized

    predicted_distance_transform_masked_ds = daisy.prepare_ds(
        'test_mito.n5',
        'predicted_distance_transform_masked',
        total_roi=binarized_daisy.roi,
        voxel_size=binarized_daisy.voxel_size,
        dtype=np.uint64)

    print("Storing predicted distance masked...")
    predicted_distance_transform_masked_ds[test_roi] = predicted_distance_transform_masked

    actual_distance_transform_masked_ds = daisy.prepare_ds(
        'test_mito.n5',
        'actual_distance_transform_masked',
        total_roi=binarized_daisy.roi,
        voxel_size=binarized_daisy.voxel_size,
        dtype=np.uint64)

    print("Storing actual distance masked...")
    actual_distance_transform_masked_ds[test_roi] = actual_distance_transform_masked



