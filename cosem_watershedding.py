from blockwise_segmentation_function import *
import funlib.segment.arrays as fsa

start_xyz = [6500, 100, 3500]
dims_xyz = [500,500,500]#[1500, 700, 1700]

test_roi = daisy.Roi((start_xyz[2]*4, start_xyz[1]*4, start_xyz[0]*4), (dims_xyz[2]*4, dims_xyz[1]*4, dims_xyz[0]*4) )

test_file = '/nrs/cosem/cosem/training/v0003.2/setup25/HeLa_Cell3_4x4x4nm/HeLa_Cell3_4x4x4nm_it600000.n5'
output_file = f'/groups/cosem/cosem/ackermand/SegmentationFromJan/new_thresh_{start_xyz[2]}_{start_xyz[1]}_{start_xyz[0]}_x_{dims_xyz[2]}_{dims_xyz[1]}_{dims_xyz[0]}.n5'
dataset = 'mito'

if __name__ == '__main__':
	for threshold in [0.9]:#[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
		array_in = daisy.open_ds(test_file, dataset)
		array_in = array_in[test_roi]

		array_out = daisy.prepare_ds(output_file,
									f'{dataset}_daisy_{threshold}',
									array_in.roi,
									voxel_size = array_in.voxel_size,
									write_size= [180*4, 180*4, 180*4],
									dtype = np.uint64)

		fsa.segment_blockwise(array_in,
							   array_out,
							   block_size = [180*4,180*4,180*4],
							   context = (200,200,200),
							   num_workers = 32,
							   segment_function = lambda array_in,roi: blockwise_segmentation_function(array_in, roi, threshold))
