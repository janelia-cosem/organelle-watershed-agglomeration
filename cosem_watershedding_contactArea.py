from blockwise_segmentation_function import *
import funlib.segment.arrays as fsa
import argparse
import os
import multiprocessing

input_file = '/nrs/cosem/cosem/training/v0003.2/setup01/HeLa_Cell3_4x4x4nm/HeLa_Cell3_4x4x4nm_it825000.n5'
output_file = '/groups/cosem/cosem/ackermand/HeLa_Cell3_4x4x4nm_setup01_it825000_results.n5'
dataset = 'mito'
num_processors = int(multiprocessing.cpu_count()/2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Watershedding and agglomeration for COSEM')
	parser.add_argument("--quantile", default=50, type=int, help="Quantile")
	args = parser.parse_args()
	quantile = args.quantile

	try:
		os.mkdir(output_file)
	except:
		pass

	file1 = open(f"{output_file}/input.txt", "w") 
	file1.write(input_file) 
	file1.close() 
	
	for threshold in [.95, .975, .99, .995, .9995]: #[0.05, 0.1, 0.5, 0.75, 0.9, 0.95, 0.99]:
		array_in = daisy.open_ds(input_file, dataset)
		
		voxel_size = array_in.voxel_size
		context_nm = (50*voxel_size[0],50*voxel_size[1],50*voxel_size[2]) #50 pixel overlap
		chunks = array_in.data.chunks
		block_size_nm = [chunks[0]*voxel_size[0],  chunks[1]*voxel_size[1], chunks[2]*voxel_size[2]]

		array_out = daisy.prepare_ds(output_file,
									f'{quantile}_{threshold}_smoothed_contactArea',
									array_in.roi,
									voxel_size = voxel_size,
									write_size= block_size_nm,
									dtype = np.uint64)

		fsa.segment_blockwise(array_in,
							   array_out,
							   block_size = block_size_nm,
							   context = context_nm,
							   num_workers = num_processors,
							   segment_function = lambda array_in,roi: blockwise_segmentation_function_using_contact_area(array_in, roi, threshold, quantile))
