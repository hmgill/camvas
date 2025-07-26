import tensorflow as tf
import numpy as np
import dill
import warnings
import time
from pathlib import Path
import random

from data_classes import * 
from image_operations import *



"""
data class : replace encoded with encoded info dict
set encoded info / metadata to data_class field 
"""


class Snapshots(tf.keras.callbacks.Callback, ImageOperations):

    
    def __init__(self,
                 fold_id,
                 images_dict,
                 samples_per_dataset = 5,
                 frequency = 5,
                 image_size = 512, 
                 output_dir = "output/snapshots",
                 random_seed = 42):
        
        super().__init__()
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        self.fold_id = fold_id

        self.frequency = int(frequency)
        
        if self.frequency <= 0:
            raise ValueError("frequency must be a positive integer.")

        # Process the dictionary of datasets
        self.set_dataset_samples(images_dict, samples_per_dataset)
        
        self.output_dir = Path(output_dir)
        
        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)


        
        
    def set_dataset_samples(self, images_dict, samples_per_dataset):
        dataset_samples = {}
        for dataset_name, images_np in images_dict.items():
            if isinstance(images_np, np.ndarray):
                # Convert to list of individual images
                all_images = [img for img in images_np]
            elif isinstance(images_np, list):
                all_images = images_np
            else:
                raise TypeError(f"images_dict['{dataset_name}'] must be a NumPy array or a list of NumPy arrays.")
            
            if not all_images:
                warnings.warn(f"PredictAndSaveCallback received an empty list of images for '{dataset_name}'.",
                              UserWarning)
                dataset_samples[dataset_name] = []
            else:
                # set images to be positive examples only
                all_images = [x for x in all_images if x.mask.path is not None]
                
                # Select random samples, but don't exceed available images
                sample_count = min(samples_per_dataset, len(all_images))
                selections = random.sample(all_images, sample_count)

                ids = [x.image.id for x in selections]
                
                images = [
                    self._apply_clahe(
                        self.read_image_or_mask(
                            x.image.path,
                            resize_dims = (512, 512)
                        )
                    ) * (1 / 255.) for x in selections
                ]

                masks = [
                    self.read_image_or_mask(
                        x.mask.path,
                        resize_dims = (512, 512),
                        as_grayscale = True
                    ) for x in selections
                ]

                dataset_samples[dataset_name] = {
                    "ids" : ids,
                    "images" : images,
                    "masks" : masks
                }
                
                print(f"Selected {sample_count} random samples from '{dataset_name}' dataset")
        self.dataset_samples = dataset_samples  

        


        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        current_epoch = epoch + 1
        if current_epoch % self.frequency == 0:
            start_time = time.time() # Optional: for timing the save operation
            
            if not any(len(samples["images"]) > 0 for samples in self.dataset_samples.values()):
                print(f"\nEpoch {current_epoch}: PredictAndSaveCallback - No images to predict on.")
                return

            print(f"\nEpoch {current_epoch}: Running prediction and saving results...")

            # Prepare data structure to store results for each dataset
            all_datasets_results = {}
            
            # Process each dataset separately
            for dataset_name, dataset in self.dataset_samples.items():
                if not dataset:
                    print(f"Epoch {current_epoch}: No images for dataset '{dataset_name}', skipping.")
                    continue
                    
                # Convert list of images to a batch for efficient prediction
                dataset_batch = np.stack(dataset["images"], axis=0)
                
                # Predict using the current state of the model
                try:
                    prediction_batch = self.model.predict(dataset_batch, verbose=0)
                except Exception as e:
                    print(f"\nEpoch {current_epoch}: Error during model prediction for '{dataset_name}': {e}")
                    continue # Skip this dataset if prediction fails
                
                # Prepare data for serialization (only for this epoch)
                dataset_results = []
                for idx, (id, image, mask) in enumerate(zip(dataset["ids"], dataset["images"], dataset["masks"])):
                    prediction = prediction_batch[idx]
                    
                    # Store necessary info to reconstruct arrays later
                    image_info = {
                        'category': 'image',
                        'bytes': image.tobytes(),
                        'dtype': str(image.dtype),
                        'shape': image.shape,
                    }

                    mask_info = {
                        'category': 'mask',
                        'bytes': mask.tobytes(),
                        'dtype': str(mask.dtype),
                        'shape': mask.shape
                    }                    

                    prediction_info = {
                        'category': 'prediction',
                        'bytes': prediction.tobytes(),
                        'dtype': str(prediction.dtype),
                        'shape': prediction.shape
                    }
                    
                    dataset_results.append({
                        "id" : id,
                        "image" : image_info,
                        "mask" : mask_info,
                        "prediction" : prediction_info
                    })
                
                # Add results for this dataset to the overall results
                all_datasets_results[dataset_name] = dataset_results
            
            # Define the output filename for this specific epoch
            output_filename = f"fold_{self.fold_id}.{current_epoch:04d}.snapshots.dill"
            output_filepath = Path(self.output_dir, output_filename)
            
            # Serialize and save to file using dill
            try:
                with open(output_filepath.as_posix(), 'wb') as f:
                    dill.dump(all_datasets_results, f)
                end_time = time.time()
                
                # Count total images saved
                total_images = sum(len(results) for results in all_datasets_results.values())
                
                print(f"Epoch {current_epoch}: Saved predictions for {total_images} images across "
                      f"{len(all_datasets_results)} datasets to '{output_filepath}'. "
                      f"Took {end_time - start_time:.2f} seconds.")
            except Exception as e:
                print(f"\nEpoch {current_epoch}: Error saving data to {output_filepath}: {e}")
