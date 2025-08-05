
import cv2
import bm3d

import numpy as np
import matplotlib.pyplot as plt

from colormap import rgb2hex, hex2rgb
from colorutils import Color



class ImageOperations():

    def __init__(self):
        pass


    def _check_if_image_in_valid_range(self, image : np.ndarray, lower_bound : float, upper_bound : float) -> bool:

        min_val = np.min(image)
        max_val = np.max(image)

        if min_val >= lower_bound and max_val <= upper_bound:
            result = True
        else:
            result = False

        return result


    def _create_template(self, image : np.ndarray, as_grayscale = False, dtype = np.uint8) -> np.ndarray:

        channels = 1 if as_grayscale == True else 3

        return np.zeros(
            (image.shape[0], image.shape[1], channels),
            dtype = dtype
        )
    def _channelwise(self, func : callable, image : np.ndarray) -> np.ndarray:

        """
        1. split image by all channels
        2. apply function to each channel
        3. merge channels back

        """

        return cv2.merge([func(channel) for channel in cv2.split(image)])



    def _apply_bm3d_denoising(self, image : np.ndarray, sigma_psd = 0.15, stage_arg = bm3d.BM3DStages.ALL_STAGES) -> np.ndarray:

        """

        applies Block Matching and 3D Filtering (BM3D) to image

        """

        def _bm3d(image_channel_data : np.ndarray) -> np.ndarray:

            nonlocal sigma_psd
            nonlocal stage_arg

            return bm3d.bm3d(
                image_channel_data,
                sigma_psd = sigma_psd,
                stage_arg = stage_arg
            )

        if _check_if_image_in_valid_range(image, 0, 1) == True:
            return self._channelwise(
                _bm3d,
                image
            )
        else:
            pass


    def _apply_clahe(self, image : np.ndarray, clip_limit = 2.0, tile_grid_size = (8,8)) -> np.ndarray:

        """

        applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to image


        parameters
        ----------
        image [req] | np.ndarray

        clip_limit | float

        tile_grid_size | tuple (n,n)


        returns
        -------
        image with clahe-adjusted channels | np.ndarray

        """


        def _clahe(image_channel_data : np.ndarray) -> np.ndarray:

            nonlocal clip_limit
            nonlocal tile_grid_size

            clahe = cv2.createCLAHE(
                clipLimit = clip_limit,
                tileGridSize = tile_grid_size
            )
            return clahe.apply(image_channel_data)

        return self._channelwise(_clahe, image)





    def _recolor(self, image : np.ndarray, color_dict : dict) -> np.ndarray:

        image_dimensions = image.shape[-1]
        if image_dimensions != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        unique_values = list(np.unique(image))
        hex_unique_values = [rgb2hex(x[-1],x[1],x[0]) for x in list(np.unique(image.reshape(-1, image.shape[2]), axis=0))]

        unique = dict(zip(unique_values, hex_unique_values))

        for k, v in unique.items():
            if v in color_dict:
                r,g,b = hex2rgb(color_dict[v])
                image[(image == k).all(axis = 2)] = [b,g,r]

        return image


    def _skeletonize(self, image : np.ndarray) -> np.ndarray:
        pass



    def _change_fundus_background_color(self, image : np.ndarray, new_color = (255,255,255)) -> np.ndarray:

        _, thresh = cv2.threshold(
            cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY),
            5,
            255,
            cv2.THRESH_BINARY
        )
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        match = np.where(np.all(thresh == [0,0,0], axis=-1))
        image[match[0], match[1]] = new_color

        return image


    def image_or_mask_sanity_checks(self):

        def wrapper(func : callable):
            def wrapper_func(*args, **kwargs):
                f = args[0]
                if not isinstance(f, pathlib.PosixPath):
                    f = pathlib.Path(f)
                file_exists = f.is_file()
                file_is_image = filetype.is_image(str(f))

                if file_exists is False:
                    _raise_error(
                        FileNotFoundError(
                            errno.ENOENT,
                            os.strerror(errno.ENOENT),
                            str(f)
                        )
                    )
                elif file_is_image is False:
                    _raise_error(
                        TypeError,
                        message = f"{str(f)} is not in supported image format"
                    )
                else:
                    return func(*args, **kwargs)
            return wrapper_func

        return wrapper



    def get_resizing_interpolation(self, original_shape : tuple, resize_shape : tuple):

        original_size = np.prod(original_shape)
        resize_size = np.prod(resize_shape)

        # shrinking original to smaller size
        if original_size > resize_size:
            interpolation = cv2.INTER_AREA
        # expanding original to larger size
        elif original_size < resize_size:
            interpolation = cv2.INTER_CUBIC
        # identical sizes
        else:
            interpolation = cv2.INTER_LINEAR

        return interpolation



    def read_image_or_mask(self, file_name : str, as_grayscale = False, resize_dims = None) -> np.ndarray:
        file_name = str(file_name)
        if as_grayscale is True:
            output = cv2.imread(file_name, 0)

            if resize_dims is not None:
                output = cv2.resize(
                    output,
                    resize_dims,
                    interpolation = cv2.INTER_NEAREST
                )
            output[output != 0] = 255
            output = np.expand_dims(output, axis = -1)

        else:
            output = cv2.imread(file_name)
            if resize_dims is not None:
                interpolation = self.get_resizing_interpolation(
                    output.shape[:2],
                    resize_dims
                )
                output = cv2.resize(
                    output,
                    resize_dims,
                    interpolation = interpolation
                )

        return output


    def grayscale_to_color(self, image, pixel_mapping):
        """
        Convert a grayscale image to a color image using pixel value to hex color mapping.

        Parameters:
        -----------
        image : numpy.ndarray
        Input grayscale image with shape (512, 512) or (512, 512, 1)
        Pixel values should be between 0 and 255
        pixel_mapping : dict
        Dictionary mapping pixel values (0-255) to hex color values
        Example: {0: '#000000', 255: '#FFFFFF', 128: '#FF0000'}

        Returns:
        --------
        numpy.ndarray
        Color image with shape (512, 512, 3) with RGB values between 0-255
        """
        # Handle different input shapes
        if image.ndim == 3 and image.shape[2] == 1:
            # Convert (512, 512, 1) to (512, 512)
            image = image.squeeze(axis=2)
        elif image.ndim != 2:
            raise ValueError("Input image must have shape (512, 512) or (512, 512, 1)")

        # Validate image dimensions
        if image.shape != (512, 512):
            raise ValueError("Input image must have dimensions 512x512")

        # Initialize output color image
        color_image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Process each pixel mapping
        for pixel_value, hex_color in pixel_mapping.items():
            # Find pixels with this value
            mask = (image == pixel_value)

            if np.any(mask):
                # Convert hex to RGB using colorutils
                color = Color(hex=hex_color)
                rgb = color.rgb

                # Convert from 0-1 range to 0-255 range
                r = int(rgb[0] * 255)
                g = int(rgb[1] * 255)
                b = int(rgb[2] * 255)

                # Apply color to matching pixels
                color_image[mask] = [r, g, b]

        # For unmapped pixels, keep them as grayscale
        unmapped_mask = np.ones(image.shape, dtype=bool)
        for pixel_value in pixel_mapping.keys():
            unmapped_mask &= (image != pixel_value)

        if np.any(unmapped_mask):
            # Convert unmapped grayscale values to RGB
            gray_values = image[unmapped_mask]
            color_image[unmapped_mask] = np.stack([gray_values, gray_values, gray_values], axis=1)

        return color_image



    def overlay(self, image1 : np.ndarray, image2 : np.ndarray, alpha = 0.6, gamma = 0):
        beta = (1.0 - alpha)
        output = cv2.addWeighted(image1, alpha, image2, beta, gamma)
        return output


    def to_range_0_255(self, image):
        # scale an image to pixel value range [0 - 255]
        return (image * 255.).astype(np.uint8)


    def to_range_0_1(self, image):
        # scale an image to pixel value range [0 - 1]
        return image * (1 / 255.) # .astype(np.uint8)
    

    def binary_threshold(self, array, threshold):
        # perform binary thresholding with a cutoff threshold value 
        return (array >= threshold).astype(int)

    
    def from_byte_string(self, metadata):
        # Convert byte string back to numpy array
        array = np.frombuffer(metadata["bytes"], dtype=metadata["dtype"])

        # Reshape the array to the original dimensions
        return array.reshape(metadata["shape"])



    def fig_to_ndarray(self, fig):
        # Draw the figure to get pixel data
        fig.canvas.draw()

        # Convert to numpy array
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        array = buf.reshape(nrows, ncols, 3)

        # clear to save memory
        plt.close(fig)

        return array


