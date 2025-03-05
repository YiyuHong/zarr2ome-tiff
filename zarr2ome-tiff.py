import numpy as np
from tifffile import TiffWriter
from utils_ome_tiff import create_ome_xml
import random
import zarr


NUMPY_FORMAT_BF_DTYPE = {'uint8': 'uint8',
                         'int8': 'int8',
                         'uint16': 'uint16',
                         'int16': 'int16',
                         'uint32': 'uint32',
                         'int32': 'int32',
                         'float32': 'float',
                         'float64': 'double'}



save_path = 'output.ome.tif'
tmp_zarr_path = 'temp.zarr'


### multi-channel channel-name
channel_names = ["DAPI", "GFP", "Cy3", "Cy5", "TxRed"]

# Define shape and chunk size
tile_size = 256
mpp = 0.25 ### set mpp metadata
shape = (len(channel_names), 30000, 20000)
chunk_size = (len(channel_names), tile_size, tile_size) 
subifds = 6 ### ome tiff pyramid level
num_cpu = 8


im = zarr.open(tmp_zarr_path, mode="w", shape=shape, dtype="uint8", chunks=chunk_size)

for _ in range(2000):
    print(_)
    # Generate a random patch
    patch = np.random.randint(0, 256, size=(len(channel_names), tile_size, tile_size), dtype=np.uint8)

    # Choose a random location in the image
    y = random.randint(0, shape[1] - tile_size)
    x = random.randint(0, shape[2] - tile_size)

    # Assign the patch to the main array
    im[:, y:y+tile_size, x:x+tile_size] = patch



nchannels = im.shape[0]
xyzct = (im.shape[2], im.shape[1], 1, im.shape[0], 1)
pixel_physical_size_xyu = mpp, mpp, "µm"
ome_xml_obj = create_ome_xml(xyzct, 'uint8', False, pixel_physical_size_xyu=pixel_physical_size_xyu, channel_names=channel_names)
ome_xml = ome_xml_obj.to_xml().encode()


def tiles_generator(data, tileshape):
    for c in range(data.shape[0]):
        for y in range(0, data.shape[1], tileshape[0]):
            for x in range(0, data.shape[2], tileshape[1]):
                yield data[c, y : y + tileshape[0], x : x + tileshape[1]]


with TiffWriter(save_path, ome=False, bigtiff=True) as tif:

        tif.write(
            tiles_generator(im, (tile_size,tile_size)),
            shape=im.shape,
            dtype=im.dtype,
            compression='jpeg',
            compressionargs={"level": 85},
            description=ome_xml,
            subifds=subifds,
            metadata=False,  # do not write tifffile metadata
            tile=(tile_size, tile_size),
            photometric='minisblack',
            resolutionunit='CENTIMETER',
            resolution=(1e4 /mpp, 1e4 / mpp),
            maxworkers=num_cpu,

        )
        for i in range(subifds):
            res = 2 ** (i + 1)
            tif.write(
                tiles_generator(im[:, ::res, ::res], (tile_size,tile_size)),
                shape=im[:, ::res, ::res].shape,
                dtype=im[:, ::res, ::res].dtype,
                compression='jpeg',
                compressionargs={"level": 85},
                subfiletype=1,
                metadata=False,
                tile=(tile_size, tile_size),
                photometric='minisblack',
                resolutionunit='CENTIMETER',
                resolution=(1e4 / mpp, 1e4 / mpp),
                maxworkers=num_cpu,
            )









