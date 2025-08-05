import pathlib
import imageio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def write_animation(array, fps:int = 1, mp4_or_gif: str = "mp4",  output_dir: str = "./output/visualize/", output_filename: str = "animation"):
    """
    Write an animation from an array to either MP4 or GIF format.

    Parameters:
    -----------
    array : numpy.ndarray
        Input array with shape (frames, height, width) or (frames, height, width, channels)
    mp4_or_gif : str
        Output format - either 'mp4' or 'gif'
    fps : int or float
        Frames per second for the animation
    output_dir : str or Path
        Directory where the output file will be saved
    output_filename : str
        Name of the output file (without extension)

    Returns:
    --------
    Path
        Path object pointing to the saved animation file
    """
    # Validate input format
    format_lower = mp4_or_gif.lower()
    if format_lower not in ['mp4', 'gif']:
        logger.error(f"Invalid format '{mp4_or_gif}'. Must be 'mp4' or 'gif'")
        raise ValueError(f"Invalid format '{mp4_or_gif}'. Must be 'mp4' or 'gif'")

    # Convert paths to Path objects
    output_dir = pathlib.Path(output_dir)

    if not output_dir.is_dir():
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Construct full output path with appropriate extension
    output_path = pathlib.Path(output_dir, f"{output_filename}.{format_lower}").as_posix()

    # Validate array
    if not isinstance(array, np.ndarray):
        logger.warning("Input is not a numpy array, attempting conversion")
        array = np.array(array)


    # Write animation
    try:
        if format_lower == 'mp4':
            # MP4 writer with ffmpeg backend
            logger.info(f"Writing MP4 with {len(array)} frames at {fps} fps")
            imageio.mimwrite(
                output_path,
                array,
                fps=fps,
                codec='libx264',  # H.264 codec for better compatibility
                quality=8,  # Quality setting (0-10, higher is better)
                pixelformat='yuv420p'  # Pixel format for compatibility
            )
        else:  # gif
            logger.info(f"Writing GIF with {len(array)} frames at {fps} fps")
            # Calculate duration per frame in seconds
            duration = 1000 / fps  # Convert to milliseconds for GIF
            imageio.mimwrite(
                output_path,
                array,
                duration=duration,
                loop=0  # 0 means infinite loop
            )

        logger.success(f"Animation saved successfully: {output_path}")
        #logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        logger.error(f"Failed to write animation: {e}")
        raise



