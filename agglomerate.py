import waterz
import daisy
import numpy as np
from scipy import ndimage

start_xyz = [6500, 100, 3500]#[6900, 100, 6600] #
dims_xyz = [1500, 700, 1700]#[500,500,500]#

#start_xyz = [6900, 100, 6600] #[6500, 100, 3500]#[6900, 100, 6600] #
#dims_xyz = [500,500,500]#[1500, 700, 1700]#

test_roi = daisy.Roi((start_xyz[2]*4, start_xyz[1]*4, start_xyz[0]*4), (dims_xyz[2]*4, dims_xyz[1]*4, dims_xyz[0]*4) )# daisy.Roi((500*4, 100*4, 5500*4), (2500*4, 500*4, 2500*4) )#daisy.Roi((6600*4, 100*4, 6900*4), (500*4, 500*4, 500*4) ) #(3500, 100, 7000), (4000, 600, 7500))

test_file = f'/groups/cosem/cosem/ackermand/SegmentationFromJan/new_thresh_{start_xyz[2]}_{start_xyz[1]}_{start_xyz[0]}_x_{dims_xyz[2]}_{dims_xyz[1]}_{dims_xyz[0]}.n5'

fragments_ds = 'fragments'

if __name__ == '__main__':
    #original method
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

    dist_ds = 'dist_max' #'dist_max'
    print("Reading fragments and distances...")
    fragments = daisy.open_ds(test_file, fragments_ds)[test_roi]
    dist_ds = daisy.open_ds(test_file, dist_ds)[test_roi]
    fragments.materialize()
    dist_ds.materialize()

    print("Creating affinity arrays...")
    affs = np.stack([dist_ds.data]*3)
    print(f"{affs.shape}")

    print("Agglomerating...")
    inverted_thresholds = 1.0 - np.array(thresholds)
    for s, t in zip(waterz.agglomerate(
        affs=affs,
        fragments=fragments.data,
        thresholds=inverted_thresholds,
        scoring_function = 1 - waterz.QuantileAffinity(50)), 
        #score = penalty. fuse if penalty is low.  
        #scoring_function = 1.0 - waterz.QuantileAffinity(90)/np.tanh( waterz.ContactArea())),#1.0 - waterz.ContactArea() ), #np.tanh( waterz.ContactArea() ) ), #TANH IS JUST A PLACEHOLDER!!!!
        thresholds):

        print(f"Storing segmentation for threshold {t}...")
        segmentation_ds = daisy.prepare_ds(
            test_file,
            'updated_segmentation_%.3f'%t,
            total_roi=fragments.roi,
            voxel_size=fragments.voxel_size,
            compressor={'id': 'gzip', 'level': 6},
            write_size=[180*4,180*4,180*4],
            dtype=np.uint64)
        segmentation_ds[test_roi] = s
