import numpy as np
from tifffile import TiffWriter, imread
import zarr
from utils_ome_tiff import create_ome_xml
from tqdm import tqdm
import ome_types
import tifffile

NUMPY_FORMAT_BF_DTYPE = {'uint8': 'uint8',
                         'int8': 'int8',
                         'uint16': 'uint16',
                         'int16': 'int16',
                         'uint32': 'uint32',
                         'int32': 'int32',
                         'float32': 'float',
                         'float64': 'double'}


def tiles_generator_translated(data, tileshape, translate_y, translate_x, channel_axis, height_axis, width_axis):
    """
    Generator that yields translated tiles from the input data.
    
    Parameters:
    -----------
    data : zarr array
        Input image data as zarr array
    tileshape : tuple
        Shape of tiles (height, width)
    translate_y : int
        Y translation offset
    translate_x : int
        X translation offset
    channel_axis : int
        Index of channel axis in data
    height_axis : int
        Index of height axis in data
    width_axis : int
        Index of width axis in data
    """
    # Get dimensions based on axis positions
    data_shape = data.shape
    nchannels = data_shape[channel_axis]
    height = data_shape[height_axis]
    width = data_shape[width_axis]
    
    tile_h, tile_w = tileshape
    
    # Calculate total number of tiles for progress bar
    tiles_y = (height + tile_h - 1) // tile_h
    tiles_x = (width + tile_w - 1) // tile_w
    total_tiles = nchannels * tiles_y * tiles_x
    
    with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
        for c in range(nchannels):
            for y in range(0, height, tile_h):
                for x in range(0, width, tile_w):

                    # Calculate actual tile size (may be smaller at edges)
                    actual_tile_h = min(tile_h, height - y)
                    actual_tile_w = min(tile_w, width - x)
                    
                    # Create output tile filled with zeros (or background value)
                    output_tile = np.zeros((actual_tile_h, actual_tile_w), dtype=data.dtype)
                    
                    # Calculate source coordinates after translation
                    src_y_start = y - translate_y
                    src_y_end = src_y_start + actual_tile_h
                    src_x_start = x - translate_x
                    src_x_end = src_x_start + actual_tile_w
                    
                    # Calculate valid source region (within bounds)
                    valid_src_y_start = max(0, src_y_start)
                    valid_src_y_end = min(height, src_y_end)
                    valid_src_x_start = max(0, src_x_start)
                    valid_src_x_end = min(width, src_x_end)
                    
                    # Calculate corresponding destination coordinates in output tile
                    dst_y_start = valid_src_y_start - src_y_start
                    dst_y_end = dst_y_start + (valid_src_y_end - valid_src_y_start)
                    dst_x_start = valid_src_x_start - src_x_start
                    dst_x_end = dst_x_start + (valid_src_x_end - valid_src_x_start)
                    
                    # Copy valid region if it exists
                    if (valid_src_y_end > valid_src_y_start and 
                        valid_src_x_end > valid_src_x_start and
                        dst_y_end > dst_y_start and 
                        dst_x_end > dst_x_start):
                        
                        # Extract source data based on axis order
                        if channel_axis == 0:  # CYX format
                            source_data = data[c, valid_src_y_start:valid_src_y_end, 
                                                valid_src_x_start:valid_src_x_end]
                        elif channel_axis == 2:  # YXC format
                            source_data = data[valid_src_y_start:valid_src_y_end, 
                                                valid_src_x_start:valid_src_x_end, c]

                        
                        # Copy the source data to the appropriate position in output tile
                        output_tile[dst_y_start:dst_y_end, 
                                    dst_x_start:dst_x_end] = source_data
                    
                    yield output_tile

                    pbar.update(1)

def translate_wsi(input_path, output_path, translate_y, translate_x, tile_size=256, subifds=6, num_cpu=8):
    """
    Translate WSI image by specified pixel offsets while maintaining same dimensions.
    
    Parameters:
    -----------
    input_path : str
        Path to input OME-TIFF file
    output_path : str
        Path to output OME-TIFF file
    translate_y : int
        Translation in Y direction (positive = down, negative = up)
    translate_x : int
        Translation in X direction (positive = right, negative = left)
    tile_size : int
        Tile size for processing
    subifds : int
        Number of pyramid levels
    num_cpu : int
        Number of CPU cores for parallel processing
    """
    
    print(f"Loading input file: {input_path}")
    
    # Open the input OME-TIFF using zarr backend for memory-efficient access
    input_zarr_group = zarr.open(imread(input_path, aszarr=True))
    
    # Access the base level (level 0) of the pyramid
    input_zarr = input_zarr_group[0]
    
    # Determine channel axis by checking the shape and metadata
    print(f"Input zarr shape: {input_zarr.shape}")
    
    # Get original metadata to determine axis order
    with tifffile.TiffFile(input_path) as tif:
        ome_metadata = tif.ome_metadata
        if ome_metadata:
            ome_xml_obj = ome_types.from_xml(ome_metadata)
            pixel_obj = ome_xml_obj.images[0].pixels
            
            # Extract physical pixel size
            phys_x = pixel_obj.physical_size_x if pixel_obj.physical_size_x else 0.25
            phys_y = pixel_obj.physical_size_y if pixel_obj.physical_size_y else 0.25
            phys_unit = pixel_obj.physical_size_x_unit if pixel_obj.physical_size_x_unit else "µm"
            


            # Extract channel names
            channel_names = []
            for channel in pixel_obj.channels:
                if channel.name:
                    channel_names.append(channel.name)
                else:
                    channel_names.append(f"Channel_{len(channel_names)}")
        else:
            # Default values if no OME metadata - assume CYX order
            phys_x, phys_y, phys_unit = 0.25, 0.25, "µm"
            channel_names = []
    


        min_dim_idx = np.argmin(input_zarr.shape)
        if min_dim_idx == 0 and input_zarr.shape[0] <= 10:  # CYX
            channel_axis = 0
            height_axis = 1
            width_axis = 2
            nchannels = input_zarr.shape[0]
            height = input_zarr.shape[1]
            width = input_zarr.shape[2]
        elif min_dim_idx == 2 and input_zarr.shape[2] <= 10:  # YXC
            channel_axis = 2
            height_axis = 0
            width_axis = 1
            height = input_zarr.shape[0]
            width = input_zarr.shape[1]
            nchannels = input_zarr.shape[2]


    
    # Create channel names if not available
    if not channel_names or len(channel_names)!=nchannels:
        channel_names = [f"Channel_{i}" for i in range(nchannels)]
    
    print(f"Detected format - Channel axis: {channel_axis}, Channels: {nchannels}, Height: {height}, Width: {width}")
    print(f"Translation: Y={translate_y}, X={translate_x}")
    print(f"Channel names: {channel_names}")
    
    # Create OME-XML metadata for output (always use CYX format for output)
    xyzct = (width, height, 1, nchannels, 1)
    pixel_physical_size_xyu = phys_x, phys_y, phys_unit
    ome_xml_obj = create_ome_xml(xyzct, NUMPY_FORMAT_BF_DTYPE[str(input_zarr.dtype)], 
                                True, pixel_physical_size_xyu=pixel_physical_size_xyu, 
                                channel_names=channel_names)
    ome_xml = ome_xml_obj.to_xml().encode()
    
    
    # Reshape data to CYX format for output consistency
    if channel_axis == 0:  # Already CYX
        output_shape = input_zarr.shape
    elif channel_axis == 2:  # YXC -> CYX
        output_shape = (nchannels, height, width)


    # Write the translated image
    print(f"Writing translated image to: {output_path}")
    
    with TiffWriter(output_path, ome=False, bigtiff=True) as tif:
        # Write base resolution
        tif.write(
            tiles_generator_translated(input_zarr, (tile_size, tile_size), translate_y, translate_x,
                                     channel_axis, height_axis, width_axis),
            shape=output_shape,
            dtype=input_zarr.dtype,
            compression='jpeg',
            compressionargs={"level": 90},
            description=ome_xml,
            subifds=subifds,
            metadata=False,
            tile=(tile_size, tile_size),
            photometric='minisblack',
            resolutionunit='CENTIMETER',
            resolution=(1e4 / phys_x, 1e4 / phys_y),
            maxworkers=num_cpu,
        )
        
        # Write pyramid levels
        for i in range(subifds):
            res = 2 ** (i + 1)
            print(f"Writing pyramid level {i+1}/{subifds} (downsampling factor: {res})")
            


            # Create downsampled view from base level if pyramid level doesn't exist
            if channel_axis == 0:  # CYX format
                downsampled = input_zarr[:, ::res, ::res]
            elif channel_axis == 2:  # YXC format
                downsampled = input_zarr[::res, ::res, :]


            # Scale translation for this resolution level
            scaled_translate_y = translate_y // res
            scaled_translate_x = translate_x // res
            
            # Reshape data to CYX format for output consistency
            if channel_axis == 0:  # Already CYX
                pyr_output_shape = downsampled.shape
            elif channel_axis == 2:  # YXC -> CYX
                pyr_output_shape = (downsampled.shape[channel_axis], downsampled.shape[height_axis], downsampled.shape[width_axis])


            
            tif.write(
                tiles_generator_translated(downsampled, (tile_size, tile_size), 
                                         scaled_translate_y, scaled_translate_x,
                                         channel_axis, height_axis, width_axis),
                shape=pyr_output_shape,
                dtype=downsampled.dtype,
                compression='jpeg',
                compressionargs={"level": 90},
                subfiletype=1,
                metadata=False,
                tile=(tile_size, tile_size),
                photometric='minisblack',
                resolutionunit='CENTIMETER',
                resolution=(1e4 / (phys_x * res), 1e4 / (phys_y * res)),
                maxworkers=num_cpu,
            )
    
    print("Translation completed successfully!")


# Example usage
if __name__ == '__main__':
    # Example parameters
    input_file = "input.ome.tiff"
    output_file = "output.ome.tiff"
    
    # Translation parameters (positive = down/right, negative = up/left)
    translate_y = 100  # Move down 100 pixels
    translate_x = 100  # Move right 100 pixels
    
    # Processing parameters
    tile_size = 256
    pyramid_levels = 6
    num_cpu = 8
    
    translate_wsi(
        input_path=input_file,
        output_path=output_file,
        translate_y=translate_y,
        translate_x=translate_x,
        tile_size=tile_size,
        subifds=pyramid_levels,
        num_cpu=num_cpu
    )
