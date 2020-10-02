from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import daisy
import numpy as np

test_roi = daisy.Roi((5500, 100, 7500), (501, 501, 501))

test_file = '/groups/cosem/cosem/data/HeLa_Cell3_4x4x4nm/HeLa_Cell3_4x4x4nm.n5'
raw_ds = '/volumes/raw'
er_ds = 'er'


if __name__ == '__main__':

    print("Opening datasets...")
    raw = daisy.open_ds(test_file, raw_ds)

    print(f"Got distances in {raw.roi}")
    raw = raw[test_roi]
    raw.materialize()

    raw_daisy_array = daisy.Array(
        data=raw.data,
        roi=test_roi,
        voxel_size=raw.voxel_size)

    raw_ds = daisy.prepare_ds(
        'test_mito.n5',
        'raw',
        total_roi=raw_daisy_array.roi,
        voxel_size=raw_daisy_array.voxel_size,
        dtype=np.uint64)

    print("Storing raw...")
    raw_ds[test_roi] = raw.data