import os
import sys
import pathlib
import pyvips
import ome_types
import colour
from scipy.spatial import distance
from matplotlib import cm
import unicodedata
import re
import itertools
import time 
import numpy as np





NUMPY_FORMAT_VIPS_DTYPE = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
    }

VIPS_FORMAT_NUMPY_DTYPE = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

NUMPY_FORMAT_BF_DTYPE = {'uint8': 'uint8',
                         'int8': 'int8',
                         'uint16': 'uint16',
                         'int16': 'int16',
                         'uint32': 'uint32',
                         'int32': 'int32',
                         'float32': 'float',
                         'float64': 'double'}
def get_n_colors(rgb, n):
    """
    Pick n most different colors in rgb. Differences based of rgb values in the CAM16UCS colorspace
    Based on https://larssonjohan.com/post/2016-10-30-farthest-points/
    """
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < rgb.max() <= 255 and np.issubdtype(rgb.dtype, np.integer):
            cam = colour.convert(rgb/255, 'sRGB', 'CAM16UCS')
        else:
            cam = colour.convert(rgb, 'sRGB', 'CAM16UCS')

    sq_D = distance.cdist(cam, cam)
    max_D = sq_D.max()
    most_dif_2Didx = np.where(sq_D == max_D)  # 2 most different colors
    most_dif_img1 = most_dif_2Didx[0][0]
    most_dif_img2 = most_dif_2Didx[1][0]
    rgb_idx = [most_dif_img1, most_dif_img2]

    possible_idx = list(range(sq_D.shape[0]))
    possible_idx.remove(most_dif_img1)
    possible_idx.remove(most_dif_img2)

    for new_color in range(2, n):
        max_d_idx = np.argmax([np.min(sq_D[i, rgb_idx]) for i in possible_idx])
        new_rgb_idx = possible_idx[max_d_idx]
        rgb_idx.append(new_rgb_idx)
        possible_idx.remove(new_rgb_idx)

    return rgb[rgb_idx]

def rgb2jch(rgb, cspace='CAM16UCS', h_rotation=0):
    jab = rgb2jab(rgb, cspace)
    jch = colour.models.Jab_to_JCh(jab)
    jch[..., 2] += h_rotation

    above_360 = np.where(jch[..., 2] > 360)
    if len(above_360[0]) > 0:
        jch[..., 2][above_360] = jch[..., 2][above_360] - 360

    return jch

def rgb2jab(rgb, cspace='CAM16UCS'):
    eps = np.finfo("float").eps
    if np.issubdtype(rgb.dtype, np.integer) and rgb.max() > 1:
        rgb01 = rgb/255.0
    else:
        rgb01 = rgb

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        jab = colour.convert(rgb01+eps, 'sRGB', cspace)

    return jab
def rgb2jch(rgb, cspace='CAM16UCS', h_rotation=0):
    jab = rgb2jab(rgb, cspace)
    jch = colour.models.Jab_to_JCh(jab)
    jch[..., 2] += h_rotation

    above_360 = np.where(jch[..., 2] > 360)
    if len(above_360[0]) > 0:
        jch[..., 2][above_360] = jch[..., 2][above_360] - 360

    return jch
def get_matplotlib_channel_colors(n_colors, name="gist_rainbow", min_lum=0.5, min_c=0.2):
    """Get channel colors using matplotlib colormaps

    Parameters
    ----------
    n_colors : int
        Number of colors needed.

    name : str
        Name of matplotlib colormap

    min_lum : float
        Minimum luminosity allowed

    min_c : float
        Minimum colorfulness allowed

    Returns
    --------
    channel_colors : ndarray
        RGB values for each of the `n_colors`

    """
    n = 200
    if n_colors > n:
        n  = n_colors
    all_colors =  cm.get_cmap(name)(np.linspace(0, 1, n))[..., 0:3]

    # Only allow bright colors #
    jch = rgb2jch(all_colors)
    all_colors = all_colors[(jch[..., 0] >= min_lum) & (jch[..., 1] >= min_c)]
    channel_colors = get_n_colors(all_colors, n_colors)
    channel_colors = (255*channel_colors).astype(np.uint8)

    return channel_colors

def remove_control_chars(s):
    """Remove control characters

    Control characters shouldn't be in some strings, like channel names.
    This will remove them

    Parameters
    ----------
    s : str
        String to check

    Returns
    -------
    control_char_removed : str
        `s` with control characters removed.

    """

    control_chars = ''.join(map(chr, itertools.chain(range(0x00,0x20), range(0x7f,0xa0))))
    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    control_char_removed = control_char_re.sub('', s)

    return control_char_removed

def create_channel(channel_id, name=None, color=None):
    """Create an ome-xml channel

    Parameters
    ----------
    channel_id : int
        Channel number

    name : str, optinal
        Channel name

    color : tuple of int
        Channel color

    Returns
    -------
    new_channel : ome_types.model.channel.Channel
        Channel object

    """

    if name is not None:
        unicode_name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore')
        decoded_name = unicode_name.decode('unicode_escape')
        decoded_name = remove_control_chars(decoded_name)

    else:
        decoded_name = None

    new_channel = ome_types.model.Channel(id=f"Channel:{channel_id}")
    if name is not None:
        new_channel.name = decoded_name
    if color is not None:

        if len(color) == 3:
            new_channel.color = tuple([*color, 1])
        elif len(color) == 4:
            if color[3] == 0:
                # color has alpha, won't be shown because it's 0
                color = tuple([*color[:3], 1])
        new_channel.color = tuple(color)

    return new_channel

def create_ome_xml(shape_xyzct, bf_dtype, is_rgb, pixel_physical_size_xyu=None, channel_names=None, colormap=None):
    """Create new ome-xmml object

    Parameters
    -------
    shape_xyzct : tuple of int
        XYZCT shape of image

    bf_dtype : str
        String format of Bioformats datatype

    is_rgb : bool
        Whether or not the image is RGB

    pixel_physical_size_xyu : tuple, optional
        Physical size per pixel and the unit.

    channel_names : list, optional
        List of channel names.

    colormap : dict, optional
        Dictionary of channel colors, where the key is the channel name, and the value the color as rgb255.
        If None (default), the channel colors from `current_ome_xml_str` will be used, if available.

    Returns
    -------
    new_ome : ome_types.model.OME
        ome_types.model.OME object containing ome-xml metadata

    """

    x, y, z, c, t = shape_xyzct
    new_ome = ome_types.OME()
    new_img = ome_types.model.Image(
        id="Image:0",
        pixels=ome_types.model.Pixels(
            id="Pixels:0",
            size_x=x,
            size_y=y,
            size_z=z,
            size_c=c,
            size_t=t,
            type=bf_dtype,
            dimension_order='XYZCT',
            metadata_only=True
        )
    )
    # "µm" "MICROMETER"
    if pixel_physical_size_xyu is not None:
        phys_x, phys_y, phys_u = pixel_physical_size_xyu
        new_img.pixels.physical_size_x = phys_x
        new_img.pixels.physical_size_x_unit = phys_u
        new_img.pixels.physical_size_y = phys_y
        new_img.pixels.physical_size_y_unit = phys_u

    if is_rgb:
        rgb_channel = ome_types.model.Channel(id='Channel:0:0', samples_per_pixel=3)
        new_img.pixels.channels = [rgb_channel]

    else:

        if channel_names is None:
            channel_names = [f"C{i}" for i in range(c)]

        default_colors = get_matplotlib_channel_colors(c)
        default_colormap = {channel_names[i]:tuple(default_colors[i]) for i in range(c)}
        if colormap is not None:
            if len(colormap) != c:
                colormap = default_colormap
                msg = "Number of colors in colormap not same as the number of channels. Using default colormap"
                print(msg)
        else:
            colormap = default_colormap

        channels = [create_channel(i, name=channel_names[i], color=colormap[channel_names[i]]) for i in range(c)]
        new_img.pixels.channels = channels

    new_ome = ome_types.model.OME()
    new_ome.images.append(new_img)

    return new_ome

def vips2bf_dtype(vips_format):
    """Get bioformats equivalent of the pyvips pixel type

    Parameters
    ----------
    vips_format : str
        Format of the pyvips.Image

    Returns
    -------
    bf_dtype : str
        String format of Bioformats datatype

    """

    np_dtype = VIPS_FORMAT_NUMPY_DTYPE[vips_format]
    bf_dtype = NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)]

    return bf_dtype


def get_shape_xyzct(shape_wh, n_channels):
    """Get image shape in XYZCT format

    Parameters
    ----------
    shape_wh : tuple of int
        Width and heigth of image

    n_channels : int
        Number of channels in the image

    Returns
    -------
    xyzct : tuple of int
        XYZCT shape of the image

    """

    xyzct = (*shape_wh, 1, n_channels, 1)
    return xyzct

def get_slide_extension(src_f):
    """Get slide format

    Parameters
    ----------
    src_f : str
        Path to slide

    Returns
    -------
    slide_format : str
        Slide format.

    """

    f = os.path.split(src_f)[1]
    if re.search(".ome.tif", f):
        format_split = -2
    else:
        format_split = -1
    slide_format = "." + ".".join(f.split(".")[format_split:])

    return slide_format


def numpy2vips(a, pyvips_interpretation=None):
    """

    """

    if a.ndim > 2:
        height, width, bands = a.shape
    else:
        height, width = a.shape
        bands = 1

    linear = a.reshape(width * height * bands)
    if linear.dtype.byteorder == ">":
        #vips seems to expect the array to be little endian, but `a` is big endian
        linear.byteswap(inplace=True)

    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      NUMPY_FORMAT_VIPS_DTYPE[a.dtype.name])

    if pyvips_interpretation is not None:
        vi = vi.copy(interpretation=pyvips_interpretation)
    return vi


def get_elapsed_time_string(elapsed_time, rounding=3):
    """Format elapsed time

    Parameters
    ----------
    elapsed_time : float
        Elapsed time in seconds

    rounding : int
        Number of decimal places to round

    Returns
    -------
    scaled_time : float
        Scaled amount of elapsed time

    time_unit : str
        Time unit, either seconds, minutes, or hours

    """

    if elapsed_time < 60:
        scaled_time = elapsed_time
        time_unit = "seconds"

    elif 60 <= elapsed_time < 60 ** 2:
        scaled_time = elapsed_time / 60
        time_unit = "minutes"

    else:
        scaled_time = elapsed_time / (60 ** 2)
        time_unit = "hours"

    scaled_time = round(scaled_time, rounding)

    return scaled_time, time_unit


def save_ome_tiff(img, dst_f, pixel_physical_size_xyu, ome_xml=None, tile_wh=1024, compression="lzw", Q=100, channel_names=None):
    """Save an image in the ome.tiff format using pyvips

    Parameters
    ---------
    img : pyvips.Image, ndarray
        Image to be saved. If a numpy array is provided, it will be converted
        to a pyvips.Image.

    ome_xml : str, optional
        ome-xml string describing image's metadata. If None, it will be createdd

    tile_wh : int
        Tile shape used to save `img`. Used to create a square tile, so `tile_wh`
        is both the width and height.

    compression : str
        Compression method used to save ome.tiff . Default is lzw, but can also
        be jpeg or jp2k. See pyips for more details.

    Q : int
        Q factor for lossy compression

    """
    compression = compression.lower()

    dst_dir = os.path.split(dst_f)[0]
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    if not isinstance(img, pyvips.vimage.Image):
        img = numpy2vips(img)

    if img.format in ["float", "double"] and compression != "lzw":
        msg = f"Image has type {img.format} but compression method {compression} will convert image to uint8. To avoid this, change compression 'lzw' "

        return None

    dst_f_extension = get_slide_extension(dst_f)
    if dst_f_extension != ".ome.tiff":
        dst_dir, out_f = os.path.split(dst_f)
        new_out_f = out_f.split(dst_f_extension)[0] + ".ome.tiff"
        new_dst_f = os.path.join(dst_dir, new_out_f)
        msg = f"{out_f} is not an ome.tiff. Changing dst_f to {new_dst_f}"
        print(msg)
        dst_f = new_dst_f

    # Get ome-xml metadata #
    xyzct = get_shape_xyzct((img.width, img.height), img.bands)
    is_rgb = img.interpretation == "srgb"
    bf_dtype = vips2bf_dtype(img.format)
    if ome_xml is None:
        # Create minimal ome-xml
        ome_xml_obj = create_ome_xml(xyzct, bf_dtype, is_rgb, pixel_physical_size_xyu=pixel_physical_size_xyu, channel_names=channel_names)
    else:
        # Verify that vips image and ome-xml match
        ome_xml_obj = ome_types.from_xml(ome_xml, parser="xmlschema")
        ome_img = ome_xml_obj.images[0].pixels
        match_dict = {"same_x": ome_img.size_x == img.width,
                      "same_y": ome_img.size_y == img.height,
                      "same_c": ome_img.size_c == img.bands,
                      "same_type": ome_img.type.name.lower() == bf_dtype
                      }

        if not all(list(match_dict.values())):
            msg = f"mismatch in ome-xml and image: {str(match_dict)}. Will create ome-xml"
            print(msg)
            ome_xml_obj = create_ome_xml(xyzct, bf_dtype, is_rgb, channel_names=channel_names)

    ome_xml_obj.creator = f"pyvips version {pyvips.__version__}"
    ome_metadata = ome_xml_obj.to_xml()

    # Save ome-tiff using vips #
    image_height = img.height
    image_bands = img.bands
    if is_rgb:
        img = img.copy(interpretation="srgb")
    else:
        img = pyvips.Image.arrayjoin(img.bandsplit(), across=1)
        img = img.copy(interpretation="b-w")

    img.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    img.set_type(pyvips.GValue.gstr_type, "image-description", ome_metadata)

    # Set up progress bar #
    bar_len = 100
    if is_rgb:
        total = 100
    else:
        total = 100*image_bands
    tic = time.time()

    save_ome_tiff.n_complete = -1
    save_ome_tiff.current_im = None
    def eval_handler(im, progress):
        if save_ome_tiff.current_im != progress.im:
            save_ome_tiff.n_complete += 1
        save_ome_tiff.current_im = progress.im
        count = save_ome_tiff.n_complete*100 + progress.percent
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        toc = time.time()
        processing_time_h = round((toc - tic)/(60), 3)

        sys.stdout.write('[%s] %s%s %s %s %s\r' % (bar, percents, '%', 'in', processing_time_h, "minutes"))
        sys.stdout.flush()

    try:
        img.set_progress(True)
        img.signal_connect("eval", eval_handler)
    except pyvips.error.Error:
        msg = "Unable to create progress bar for pyvips. May need to update libvips to >= 8.11"
        print(msg)

    print(f"saving {dst_f} ({img.width} x {image_height} and {image_bands} channels)")

    # Write image #
    tile_wh = tile_wh - (tile_wh % 16)  # Tile shape must be multiple of 16
    if np.any(np.array(xyzct[0:2]) < tile_wh):
        # Image is smaller than the tile #
        min_dim = min(xyzct[0:2])
        tile_wh = int((min_dim - min_dim % 16))
    if tile_wh < 16:
        tile_wh = 16

    print("")

    lossless = Q == 100
    rgbjpeg = compression in ["jp2k", "jpeg"] and img.interpretation == "srgb"
    img.tiffsave(dst_f, compression=compression, tile=True,
                 tile_width=tile_wh, tile_height=tile_wh,
                 pyramid=True, subifd=True, bigtiff=True,
                 lossless=lossless, Q=Q, rgbjpeg=rgbjpeg)

    # Print total time to completion #
    toc = time.time()
    processing_time_seconds = toc-tic
    processing_time, processing_time_unit = get_elapsed_time_string(processing_time_seconds)

    bar = '=' * bar_len
    sys.stdout.write('[%s] %s%s %s %s %s\r' % (bar, 100.0, '%', 'in', processing_time, processing_time_unit))
    sys.stdout.flush()
    sys.stdout.write('\nComplete\n')
    print("")


if __name__ == '__main__':


    image = np.ones(1000,1000,3)
    save_path = "dummy.ome.tiff"
    pixel_physical_size_xyu = 0.25, 0.25, "µm"
    save_ome_tiff(image, save_path, pixel_physical_size_xyu, ome_xml=None, tile_wh=256, compression="jpeg", Q=85)



