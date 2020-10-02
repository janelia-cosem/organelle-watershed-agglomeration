from blockwise_segmentation_function import *
import funlib.segment.arrays as fsa

test_file = '/nrs/cosem/cosem/training/v0003.2/setup25/HeLa_Cell3_4x4x4nm/HeLa_Cell3_4x4x4nm_it600000.n5'
output_file = f'/groups/cosem/cosem/ackermand/SegmentationFromJan/full_rescaled.n5'
dataset = 'mito'

if __name__ == '__main__':
	for threshold in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
		array_in = daisy.open_ds(test_file, dataset)

		array_out = daisy.prepare_ds(output_file,
									f'50_{threshold}',
									array_in.roi,
									voxel_size = array_in.voxel_size,
									write_size= [180*4, 180*4, 180*4],
									dtype = np.uint64)

		fsa.segment_blockwise(array_in,
							   array_out,
							   block_size = [180*4,180*4,180*4],
							   context = (200,200,200),
							   num_workers = 32,
							   segment_function = lambda array_in,roi: blockwise_segmentation_function(array_in, roi, threshold, 50))
