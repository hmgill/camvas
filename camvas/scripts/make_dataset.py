#!/usr/bin/env python3
"""
Dataset CSV Generator

Creates a dataset CSV file by matching images to masks across multiple data sources.
Each data source can specify an image directory, optional mask directory, and role.
"""

import argparse
import csv
import imghdr
from pathlib import Path
from typing import List, Dict, Optional, Set
import sys
from loguru import logger


def is_valid_image(file_path: Path) -> bool:
    """
    Check if a file is a valid image using imghdr.
    
    Args:
        file_path: Path to the file to check
    
    Returns:
        True if the file is a valid image, False otherwise
    """
    try:
        return imghdr.what(file_path) is not None
    except (OSError, IOError):
        return False


def get_image_files(image_dir: Path, extensions: Set[str] = None) -> Dict[str, Path]:
    """
    Get all image files from a directory and return a dict mapping stem names to full paths.
    
    Args:
        image_dir: Path to the image directory
        extensions: Set of valid image extensions (default: common image formats)
    
    Returns:
        Dictionary mapping filename stems to full file paths
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    image_files = {}
    if not image_dir.exists():
        logger.warning(f"Image directory does not exist: {image_dir}")
        return image_files
    
    for file_path in image_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # Verify it's actually a valid image file
            if not is_valid_image(file_path):
                logger.warning(f"Skipping invalid image file: {file_path}")
                continue
                
            stem = file_path.stem
            if stem in image_files:
                logger.warning(f"Duplicate image stem '{stem}' found in {image_dir}")
            image_files[stem] = file_path
    
    return image_files


def get_mask_files(mask_dir: Optional[Path], extensions: Set[str] = None) -> Dict[str, Path]:
    """
    Get all mask files from a directory and return a dict mapping stem names to full paths.
    
    Args:
        mask_dir: Path to the mask directory (can be None)
        extensions: Set of valid mask extensions (default: common image formats)
    
    Returns:
        Dictionary mapping filename stems to full file paths
    """
    if mask_dir is None:
        return {}
    
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    mask_files = {}
    if not mask_dir.exists():
        logger.warning(f"Mask directory does not exist: {mask_dir}")
        return mask_files
    
    for file_path in mask_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # Verify it's actually a valid image file
            if not is_valid_image(file_path):
                logger.warning(f"Skipping invalid mask file: {file_path}")
                continue
                
            stem = file_path.stem
            if stem in mask_files:
                logger.warning(f"Duplicate mask stem '{stem}' found in {mask_dir}")
            mask_files[stem] = file_path
    
    return mask_files


def process_data_source(dataset_name: str, image_dir: str, mask_dir: Optional[str], role: str) -> List[Dict[str, str]]:
    """
    Process a single data source and return list of image-mask pairs.
    
    Args:
        dataset_name: Name of the dataset
        image_dir: Path to image directory
        mask_dir: Path to mask directory (optional)
        role: Role assignment ("train", "benchmark", "predict")
    
    Returns:
        List of dictionaries containing image, mask, role, and dataset information
    """
    image_path = Path(image_dir)
    mask_path = Path(mask_dir) if mask_dir else None
    
    # Get all image and mask files
    image_files = get_image_files(image_path)
    mask_files = get_mask_files(mask_path)
    
    if not image_files:
        logger.warning(f"No valid image files found in {image_path}")
        return []
    
    # Create dataset entries and sort by image filename
    dataset_entries = []
    matched_masks = 0
    
    # Sort by stem (filename without extension) for consistent ordering
    for stem in sorted(image_files.keys()):
        id = stem
        image_file = image_files[stem]
        mask_file = mask_files.get(stem)
        
        entry = {
            'id' : str(id),
            'image': str(image_file),
            'mask': str(mask_file) if mask_file else 'NA',
            'role': role,
            'dataset': dataset_name
        }
        
        dataset_entries.append(entry)
        
        if mask_file:
            matched_masks += 1
    
    logger.info(f"Processed {len(dataset_entries)} images from {image_path}")
    logger.info(f"  - Matched masks: {matched_masks}/{len(dataset_entries)}")
    logger.info(f"  - Role: {role}")
    logger.info(f"  - Dataset: {dataset_name}")
    
    return dataset_entries


def validate_role(role: str) -> str:
    """Validate that role is one of the accepted values."""
    valid_roles = {'train', 'benchmark', 'predict'}
    if role not in valid_roles:
        raise argparse.ArgumentTypeError(
            f"Role must be one of {valid_roles}, got '{role}'"
        )
    return role


def parse_data_source(arg_string: str) -> tuple:
    """
    Parse a data source argument string.
    
    Expected format: "dataset_name,image_dir[,mask_dir],role"
    If mask_dir is omitted: "dataset_name,image_dir,,role"
    
    Args:
        arg_string: String containing data source information
    
    Returns:
        Tuple of (dataset_name, image_dir, mask_dir, role)
    """
    parts = arg_string.split(',')
    
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            "Data source must be in format: 'dataset_name,image_dir[,mask_dir],role'"
        )
    
    if len(parts) == 3:
        # Format: "dataset_name,image_dir,role"
        dataset_name, image_dir, role = parts
        mask_dir = None
    elif len(parts) == 4:
        # Format: "dataset_name,image_dir,mask_dir,role" or "dataset_name,image_dir,,role"
        dataset_name, image_dir, mask_dir, role = parts
        mask_dir = mask_dir if mask_dir.strip() else None
    else:
        raise argparse.ArgumentTypeError(
            "Too many comma-separated values in data source"
        )
    
    # Validate role
    role = validate_role(role.strip())
    
    return dataset_name.strip(), image_dir.strip(), mask_dir.strip() if mask_dir else None, role


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset CSV file by matching images to masks across multiple data sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Single data source with masks
        python3 make_dataset.py -d "dataset1,/path/to/images/,/path/to/masks/,train" -o dataset.csv
        
        # Single data source without masks
        python3 make_dataset.py -d "dataset2,/path/to/images/,,predict" -o dataset.csv
        
        """
    )
    
    parser.add_argument(
        '-d', '--data-source',
        dest='data_sources',
        action='append',
        required=True,
        type=parse_data_source,
        help='Data source in format: "dataset_name,image_dir[,mask_dir],role". Use empty mask_dir for no masks: "dataset_name,image_dir,,role"'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--image-extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'],
        help='Valid image file extensions (default: jpg, jpeg, png, bmp, tiff, tif, webp)'
    )
    
    args = parser.parse_args()
    
    # Convert extensions to lowercase set
    extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                  for ext in args.image_extensions}
    
    # Process all data sources
    all_entries = []
    
    logger.info(f"Processing {len(args.data_sources)} data source(s)...")
    
    for i, (dataset_name, image_dir, mask_dir, role) in enumerate(args.data_sources, 1):
        logger.info(f"--- Data Source {i} ---")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Image directory: {image_dir}")
        logger.info(f"Mask directory: {mask_dir if mask_dir else 'None'}")
        logger.info(f"Role: {role}")
        
        entries = process_data_source(dataset_name, image_dir, mask_dir, role)
        all_entries.extend(entries)
    
    # Write CSV file
    if not all_entries:
        logger.error("No valid entries found across all data sources.")
        sys.exit(1)
    
    logger.info(f"Writing {len(all_entries)} entries to {args.output}")
    
    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'image', 'mask', 'role', 'dataset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(all_entries)
    
    # Print summary
    logger.info("--- Summary ---")
    role_counts = {}
    mask_counts = {'with_mask': 0, 'without_mask': 0}
    
    for entry in all_entries:
        role = entry['role']
        role_counts[role] = role_counts.get(role, 0) + 1
        
        if entry['mask'] != 'NA':
            mask_counts['with_mask'] += 1
        else:
            mask_counts['without_mask'] += 1
    
    logger.info(f"Total entries: {len(all_entries)}")
    logger.info(f"Entries with masks: {mask_counts['with_mask']}")
    logger.info(f"Entries without masks: {mask_counts['without_mask']}")
    
    for role, count in sorted(role_counts.items()):
        logger.info(f"{role.capitalize()} entries: {count}")
    
    logger.success(f"Dataset CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
