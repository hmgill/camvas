"""
Training callbacks for CAMVAS vessel segmentation.

This module contains custom Keras callbacks for training monitoring,
snapshot saving, and progress visualization.
"""

import time
import random
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import dill
import numpy as np
import tensorflow as tf
from loguru import logger

# Local imports
from ..data.classes import Datapoint
from ..processing.image_ops import ImageOperations


class Snapshots(tf.keras.callbacks.Callback, ImageOperations):
    """
    Callback to save model predictions during training for visualization.
    
    This callback periodically saves model predictions on a subset of data
    to monitor training progress and create visualizations/animations.
    """
    
    def __init__(
        self,
        fold_id: str,
        images_dict: Dict[str, List[Datapoint]],
        samples_per_dataset: int = 5,
        frequency: int = 5,
        image_size: int = 512,
        output_dir: str = "../../output/snapshots",
        random_seed: int = 42,
        **kwargs
    ):
        """
        Initialize the Snapshots callback.
        
        Args:
            fold_id: Identifier for the current fold
            images_dict: Dictionary mapping dataset names to lists of datapoints
            samples_per_dataset: Number of samples to track per dataset
            frequency: How often (in epochs) to save snapshots
            image_size: Size to resize images to
            output_dir: Directory to save snapshots
            random_seed: Random seed for reproducible sampling
            **kwargs: Additional arguments for parent classes
        """
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        self.fold_id = fold_id
        self.frequency = int(frequency)
        self.image_size = image_size
        self.output_dir = Path(output_dir)
        
        # Validate frequency
        if self.frequency <= 0:
            raise ValueError("Frequency must be a positive integer")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and sample datasets
        self.dataset_samples = self._prepare_dataset_samples(
            images_dict, samples_per_dataset
        )
        
        logger.info(f"Initialized Snapshots callback for fold {fold_id}")
        logger.info(f"Tracking {sum(len(samples['images']) for samples in self.dataset_samples.values())} samples")
    
    def _prepare_dataset_samples(
        self, 
        images_dict: Dict[str, List[Datapoint]], 
        samples_per_dataset: int
    ) -> Dict[str, Dict[str, List]]:
        """
        Prepare and sample datasets for snapshot tracking.
        
        Args:
            images_dict: Dictionary of dataset names to datapoint lists
            samples_per_dataset: Number of samples per dataset
            
        Returns:
            Dictionary with processed samples for each dataset
        """
        dataset_samples = {}

        for dataset_name, datapoints in images_dict.items():
            
            # Convert to list if numpy array
            if isinstance(datapoints, np.ndarray):
                datapoints = datapoints.tolist()
            
            # Filter to only positive examples (with masks)
            positive_examples = [
                dp for dp in datapoints 
                if dp.mask and dp.mask.path is not None
            ]
            
            if not positive_examples:
                logger.warning(f"No positive examples found for dataset '{dataset_name}'")
                dataset_samples[dataset_name] = {
                    "ids": [],
                    "images": [],
                    "masks": []
                }
                continue
            
            # Sample random examples
            sample_count = min(samples_per_dataset, len(positive_examples))
            selected_samples = random.sample(positive_examples, sample_count)
            
            # Process images and masks
            ids = [dp.id for dp in selected_samples]
            images = []
            masks = []
            
            for dp in selected_samples:
                try:
                    # Read and process image
                    image = self.read_image_or_mask(
                        dp.image.path,
                        resize_dims=(self.image_size, self.image_size)
                    )
                    image = self._apply_clahe(image)
                    image = image * (1 / 255.0)  # Normalize to [0, 1]
                    images.append(image)
                    
                    # Read and process mask
                    mask = self.read_image_or_mask(
                        dp.mask.path,
                        resize_dims=(self.image_size, self.image_size),
                        as_grayscale=True
                    )
                    masks.append(mask)
                    
                except Exception as e:
                    logger.error(f"Error processing sample {dp.id}: {e}")
                    continue
            
            dataset_samples[dataset_name] = {
                "ids": ids,
                "images": images,
                "masks": masks
            }
            
            logger.info(f"Selected {len(images)} samples from '{dataset_name}' dataset")
        
        return dataset_samples
    
    def _serialize_array(self, array: np.ndarray, category: str) -> Dict[str, Any]:
        """
        Serialize numpy array for storage.
        
        Args:
            array: Numpy array to serialize
            category: Category of the array (image, mask, prediction)
            
        Returns:
            Dictionary with serialized array metadata
        """
        return {
            'category': category,
            'bytes': array.tobytes(),
            'dtype': str(array.dtype),
            'shape': array.shape,
        }
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dictionary with metric results
        """
        current_epoch = epoch + 1
        
        # Check if we should save snapshots this epoch
        if current_epoch % self.frequency != 0:
            return
        
        # Check if we have any samples to process
        total_samples = sum(
            len(samples["images"]) 
            for samples in self.dataset_samples.values()
        )
        
        if total_samples == 0:
            logger.warning(f"Epoch {current_epoch}: No samples to predict on")
            return
        
        start_time = time.time()
        logger.info(f"Epoch {current_epoch}: Generating snapshots...")
        
        # Store results for all datasets
        all_datasets_results = {}
        
        # Process each dataset
        for dataset_name, dataset in self.dataset_samples.items():
            if not dataset["images"]:
                logger.debug(f"Skipping empty dataset '{dataset_name}'")
                continue
            
            try:
                # Convert list of images to batch
                dataset_batch = np.stack(dataset["images"], axis=0)
                
                # Get model predictions
                predictions = self.model.predict(dataset_batch, verbose=0)
                
                # Process each sample in the batch
                dataset_results = []
                for idx, (sample_id, image, mask) in enumerate(
                    zip(dataset["ids"], dataset["images"], dataset["masks"])
                ):
                    prediction = predictions[idx]
                    
                    # Serialize data for storage
                    sample_data = {
                        "id": sample_id,
                        "image": self._serialize_array(image, 'image'),
                        "mask": self._serialize_array(mask, 'mask'),
                        "prediction": self._serialize_array(prediction, 'prediction')
                    }
                    
                    dataset_results.append(sample_data)
                
                all_datasets_results[dataset_name] = dataset_results
                logger.debug(f"Processed {len(dataset_results)} samples from '{dataset_name}'")
                
            except Exception as e:
                logger.error(f"Error processing dataset '{dataset_name}': {e}")
                continue
        
        # Save results to file
        if all_datasets_results:
            self._save_snapshots(current_epoch, all_datasets_results, start_time)
        else:
            logger.warning(f"Epoch {current_epoch}: No snapshots generated")
    
    def _save_snapshots(
        self, 
        epoch: int, 
        results: Dict[str, List[Dict]], 
        start_time: float
    ):
        """
        Save snapshot results to file.
        
        Args:
            epoch: Current epoch number
            results: Dictionary with results for all datasets
            start_time: Time when snapshot generation started
        """
        # Create filename
        filename = f"fold_{self.fold_id}.{epoch:04d}.snapshots.dill"
        filepath = self.output_dir / filename
        
        try:
            # Save using dill
            with open(filepath, 'wb') as f:
                dill.dump(results, f)
            
            # Calculate timing and statistics
            end_time = time.time()
            duration = end_time - start_time
            total_images = sum(len(dataset_results) for dataset_results in results.values())
            
            logger.success(
                f"Epoch {epoch}: Saved {total_images} snapshots across "
                f"{len(results)} datasets to '{filename}' ({duration:.2f}s)"
            )
            
        except Exception as e:
            logger.error(f"Error saving snapshots to {filepath}: {e}")
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the start of training."""
        logger.info(f"Starting snapshot tracking for fold {self.fold_id}")
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        logger.info(f"Completed snapshot tracking for fold {self.fold_id}")


class MetricsLogger(tf.keras.callbacks.Callback):
    """
    Enhanced metrics logging callback using loguru.
    """
    
    def __init__(self, log_frequency: int = 1):
        """
        Initialize metrics logger.
        
        Args:
            log_frequency: How often (in epochs) to log metrics
        """
        super().__init__()
        self.log_frequency = log_frequency
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics at the end of each epoch."""
        if (epoch + 1) % self.log_frequency != 0:
            return
        
        if logs:
            epoch_num = epoch + 1
            
            # Separate training and validation metrics
            train_metrics = {k: v for k, v in logs.items() if not k.startswith('val_')}
            val_metrics = {k: v for k, v in logs.items() if k.startswith('val_')}
            
            # Log training metrics
            if train_metrics:
                train_msg = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
                logger.info(f"Epoch {epoch_num} - Train: {train_msg}")
            
            # Log validation metrics
            if val_metrics:
                val_msg = " | ".join([f"{k[4:]}: {v:.4f}" for k, v in val_metrics.items()])
                logger.info(f"Epoch {epoch_num} - Val: {val_msg}")


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Training callbacks module loaded successfully")
