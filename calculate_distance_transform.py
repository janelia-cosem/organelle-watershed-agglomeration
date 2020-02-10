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
    #dist_er = daisy.open_ds(test_file, er_ds)
    dist_mitos = daisy.open_ds(test_file, mito_ds)

    print(f"Got distances in {dist_mitos.roi}")
    #dist_er = dist_er[test_roi]
    dist_mitos = dist_mitos[test_roi]

    print("Reading into memory...")
    #dist_er.materialize()
    dist_mitos.materialize()
    
    print("Binarize...")
    binarized = dist_mitos.data>=127
    
    print("Calculate distance transform...")
    distance_transform = distance_transform_edt(binarized)
    distance_transform *= 4 #nm
    distance_transform = 128*np.tanh(distance_transform/50)+127
    distance_transform = np.ma.array(distance_transform, mask=binarized)
    
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

    distance_transform_ds = daisy.prepare_ds(
        'test_mito.n5',
        'distance_transform',
        total_roi=binarized_daisy.roi,
        voxel_size=binarized_daisy.voxel_size,
        dtype=np.uint64)

    print("Storing distance transform...")
    distance_transform_ds[test_roi] = distance_transform
    
