from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import daisy
import numpy as np
import copy

start_xyz = [7500,100, 3500]#[6500, 100, 3500]#[6900, 100, 6600] #
dims_xyz = [1500, 700, 1700]#[500,500,500]#

test_roi = daisy.Roi((start_xyz[2]*4, start_xyz[1]*4, start_xyz[0]*4), (dims_xyz[2]*4, dims_xyz[1]*4, dims_xyz[0]*4) )# daisy.Roi((500*4, 100*4, 5500*4), (2500*4, 500*4, 2500*4) )#daisy.Roi((6600*4, 100*4, 6900*4), (500*4, 500*4, 500*4) ) #(3500, 100, 7000), (4000, 600, 7500))

test_file = '/nrs/cosem/cosem/training/v0003.2/setup25/HeLa_Cell3_4x4x4nm/HeLa_Cell3_4x4x4nm_it600000.n5'
output_file = f'/groups/cosem/cosem/ackermand/SegmentationFromJan/new_thresh_{start_xyz[2]}_{start_xyz[1]}_{start_xyz[0]}_x_{dims_xyz[2]}_{dims_xyz[1]}_{dims_xyz[0]}.n5'
mito_ds = 'mito'

def find_seeds(dist, sigma=3.0):

    smoothed = gaussian_filter(dist, sigma=sigma)
    return label(peak_local_max(smoothed, indices=False))[0]

def normalize_distances(array):

    array.data = array.data.astype(np.float32)/255.0
    array.data -= 126/255.0
    array.data[array.data<0] = 0
    array.data /= 129/255.0

def calculate_dist_in_voxels(array):
    array.data[array.data==255] = 254 #To prevent infinity
    array.data = np.arctanh( (array.data-127.0)/128.0 ) *50.0/4
    print(np.amax(array.data))
    array.data[array.data<0] = 0 

if __name__ == '__main__':

    print("Opening datasets...")
    dist_mitos = daisy.open_ds(test_file, mito_ds)

    print(f"Got distances in {dist_mitos.roi}")
    dist_mitos = dist_mitos[test_roi]

    print("Reading into memory...")
    dist_mitos.materialize()

    print("Normalizing distances...")
    normalize_distances(dist_mitos)

    # TODO: include all classes of the same level in the hierarchy
    print("Computing max distances...")
    dist_max = np.abs(dist_mitos.data)
    
    print("Finding seeds...")
    seeds = find_seeds(dist_max)
    print(f"Num seeds: {np.count_nonzero(seeds)}")

    print("Running watershed...")
    print(f"dist_max: {dist_max.shape}")
    print(f"seeds: {seeds.shape}")
    fragments = watershed(-dist_max, seeds, mask = (dist_max.astype(bool))).astype(np.uint64)

    fragments = daisy.Array(
        data=fragments,
        roi=test_roi,
        voxel_size=dist_mitos.voxel_size)

    fragments_ds = daisy.prepare_ds(
        output_file,
        'fragments',
        total_roi=fragments.roi,
        voxel_size=fragments.voxel_size,
        dtype=np.uint64)

    print("Storing fragments...")
    fragments_ds[test_roi] = fragments

    dist_max_ds = daisy.prepare_ds(
        output_file,
        'dist_max',
        total_roi=fragments.roi,
        voxel_size=fragments.voxel_size,
        dtype=np.float32)

    print("Storing dist_max...")
    dist_max_ds[test_roi] = dist_max


    # ## due to memory issue, read in again
    # print("Opening datasets...")
    # dist_mitos = daisy.open_ds(test_file, mito_ds)

    # print(f"Got distances in {dist_mitos.roi}")
    # dist_mitos = dist_mitos[test_roi]

    # print("Reading into memory...")
    # dist_mitos.materialize()

    # print("Calculate dist in voxels...")
    # calculate_dist_in_voxels(dist_mitos)

    # dist_mitos_ds = daisy.prepare_ds(
    #     output_file,
    #     'dist_mitos_in_voxels',
    #     total_roi=fragments.roi,
    #     voxel_size=fragments.voxel_size,
    #     dtype=np.float32)

    # print("Storing dist_in_voxels")
    # dist_mitos_ds[test_roi] = dist_mitos