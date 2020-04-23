from blockwise_segmentation_function import *
import funlib.segment.arrays as fsa
import argparse

input_file = '/nrs/cosem/cosem/training/v0003.2/setup25/HeLa_Cell3_4x4x4nm/HeLa_Cell3_4x4x4nm_it600000.n5'
output_file = '/groups/cosem/cosem/ackermand/SegmentationFromJan/full_rescaled.n5'
dataset = 'mito'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Watershedding and agglomeration for COSEM')
	parser.add_argument("--quantile", default=50, type=int, help="Quantile")
	args = parser.parse_args()
	quantile = args.quantile

	for threshold in [0.5]:
		array_in = daisy.open_ds(input_file, dataset)

		array_out = daisy.prepare_ds(output_file,
									f'{quantile}_{threshold}',
									array_in.roi,
									voxel_size = array_in.voxel_size,
									write_size= [180*4, 180*4, 180*4],
									dtype = np.uint64)

		fsa.segment_blockwise(array_in,
							   array_out,
							   block_size = [180*4,180*4,180*4],
							   context = (200,200,200),
							   num_workers = 48,
							   segment_function = lambda array_in,roi: blockwise_segmentation_function(array_in, roi, threshold, quantile))
