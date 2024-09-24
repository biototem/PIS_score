import gc
from tifffile import TiffWriter
import cv2


def img_to_tif(tiff_slide,data,output_path_pp):

    # 一些svs定义
    svs_desc = 'Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}'
    label_desc = 'Aperio Image Library Fake\nlabel {W}x{H}'
    macro_desc = 'Aperio Image Library Fake\nmacro {W}x{H}'

    default_metadata = {'MPP': float(tiff_slide.properties['tiffslide.mpp-x']),'AppMag': int(tiff_slide.properties['aperio.AppMag'])}
    pixelsize = float(default_metadata['MPP'])

    # 缩略图
    thumbnail_im =  cv2.resize(data,(int(data.shape[1]/8),int(data.shape[0]/8)))
    label_im = cv2.resize(data,(int(data.shape[1]/8),int(data.shape[0]/8)))
    macro_im = cv2.resize(data,(int(data.shape[1]/8),int(data.shape[0]/8)))

    subresolutions = 4

    with TiffWriter(output_path_pp, bigtiff=True) as tif:
        metadata={
            'axes': 'YX',
            'SignificantBits': 10,
            'Channel': {'Name': ['Channel 1', 'Channel 2']},
            'TimeIncrement': 0.1,
            'TimeIncrementUnit': 's',
            'PhysicalSizeX': pixelsize,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixelsize,
            'PhysicalSizeYUnit': 'µm',
        }
        options = dict(
            photometric='rgb',
            tile=(256, 256),
            compression='jpeg',
            resolutionunit='CENTIMETER'
        )
        tif.write(
            data,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options
        )
        # save pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
        for level in range(subresolutions):
            mag = 2**(level + 1)
            tif.write(
                data[..., ::mag, ::mag, :],##rgb
                subfiletype=1,
                resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
                **options
            )
            if level == 0:
                # 写入缩略图
                tif.write(data=thumbnail_im, description='',**options)

        # 写入标签图
        tif.write(data=label_im, subfiletype=1, description=label_desc.format(W=label_im.shape[1], H=label_im.shape[0]),
                  **options)
        # 写入宏观图
        tif.write(data=macro_im, subfiletype=9, description=macro_desc.format(W=macro_im.shape[1], H=macro_im.shape[0]),
                  **options)

        tif.close()
    try:
        del thumbnail_im, label_im,macro_im,data
        gc.collect()
        del thumbnail_im, label_im,macro_im,data
        gc.collect()
        del thumbnail_im, label_im,macro_im,data
        gc.collect()
        del thumbnail_im, label_im,macro_im,data
        gc.collect()
        del thumbnail_im, label_im,macro_im,data
        gc.collect()
    except:
        pass

