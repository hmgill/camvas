import pathlib 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from colorutils import Color

from animate import * 
from config import *
from utils import *
from model import *


class VisualizeSnapshots(Utils):


    def __init__(self, config, snapshot_dir = "./output/snapshots/"):

        self.config = config
        self.snapshot_dir = snapshot_dir
         
    

    def create_three_panel_fig(self, image, mask, prediction, alpha=0.3, legend_dict=None, title=""):
        """
        Create a matplotlib figure with three panels showing image, mask overlay, and prediction overlay.
        
        Parameters:
        - image: numpy array of the base image
        - mask: numpy array of the mask
        - prediction: numpy array of the prediction
        - opacity: float, opacity for overlays (default 0.5)
        - legend_dict: dict with format {'class': '', 'hex_value': ''}
        - title: string, overall figure title
        
        Returns:
        - fig: matplotlib figure object
        """
    
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Original image
        axes[0].imshow(image)
        axes[0].set_title('Image')
    
        # Panel 2: Ground Truth Mask
        mask = self.grayscale_to_color(mask, {255:"#148ef6"})
        mask = self.overlay(image, mask, alpha = alpha)
        axes[1].imshow(mask)
        axes[1].set_title('Ground Truth Mask')
        
        # Panel 3: Image with prediction overlay
        prediction = self.grayscale_to_color(prediction, {255:"#148ef6"})
        prediction = self.overlay(image, prediction, alpha = alpha)
        axes[2].imshow(prediction)
        axes[2].set_title('Predicted Mask')
    
        # Remove ticks and labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
    
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16, y=0.95)

        
        # Create legend if provided
        if legend_dict:
            legend_elements = []
            for class_name, hex_color in legend_dict.items():
                legend_elements.append(patches.Rectangle((0, 0),
                                                         1,
                                                         1, 
                                                         facecolor = hex_color, 
                                                         label=class_name))
        
                fig.legend(handles=legend_elements, 
                           loc='lower center', 
                           bbox_to_anchor=(0.5, 0.02),
                           ncol=len(legend_dict),
                           frameon=False
                )
    
    
        return fig    
    
    
    def restore_data(self, metadata, threshold = 0.25):

        array = self.from_byte_string(metadata)

        if metadata["category"] in ["image", "mask"]:
            # normalize to pixel value range [0 - 255]
            array = self.to_range_0_255(array)
            if metadata["category"] == 'mask':
                array = self.to_range_0_255(array)
            return array

        elif metadata["category"] == "prediction":
            array = self.binary_threshold(array, threshold)
            array = self.to_range_0_255(array)

            return array

        
        
    def get_fold_data(self):

        progress_files = list(
            pathlib.Path(self.snapshot_dir).glob("*.snapshots.dill")
        )

        data = []
        for f in progress_files:

            split = f.stem.split(".")

            fold_id = split[0]
            epoch = split[1]

            data.append({
                "fold_id" : fold_id,
                "epoch" : epoch,
                "path" : f
            })
        

        fold_ids = {x["fold_id"] for x in data}

        output = []
        for fold_id in fold_ids:
            file_matches = []
            for datapoint in data:
                if fold_id == datapoint["fold_id"]:
                    file_matches.append(datapoint)

            file_matches = sorted(file_matches, key=lambda x: x['epoch'])
            
            output.append({
                "id" : fold_id,
                "matches" : file_matches
            })
            
        return output 


    
        

    def main(self):

        """
        --- DILL FILES ---
        """
        
        """
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops = tf.distribute.ReductionToOneDevice()
        )
        
        with strategy.scope():
        """
        # read progress dill files for each fold
        fold_data = self.get_fold_data()


        for fold in fold_data:

            snapshot_data = {}

            for match in fold["matches"]:

                epoch = match["epoch"]
                
                snapshot_groups = self.read_dill(
                    match["path"].as_posix()
                )

                for snapshot_group, snapshots in snapshot_groups.items():

                    if snapshot_group not in snapshot_data:
                        snapshot_data[snapshot_group] = {}

                        
                    for snapshot in snapshots:

                        if snapshot['id'] not in snapshot_data[snapshot_group]:
                            snapshot_data[snapshot_group][snapshot['id']] = []

                            
                        
                        # snapshot id: fold - group - image id - epoch  
                        snapshot_id = f"{fold['id']}_{snapshot_group}_{snapshot['id']}_{epoch}"
                        #print(snapshot_id)
                        
                        image = self.restore_data(snapshot["image"])
                        mask = self.restore_data(snapshot["mask"])
                        prediction = self.restore_data(snapshot["prediction"])

                        # swap channels from BGR to RGB format
                        image = image[:, :, ::-1]
                        mask = mask[:, :, ::-1]
                        prediction = prediction[:, :, ::-1]


                        # create snapshot figure
                        image_mask_pred = self.create_three_panel_fig(
                            image,
                            mask,
                            prediction
                        )

                        
                        # create probability heatmap 
                        probability_heatmap, _ = self.make_probability_heatmap(
                            image,
                            prediction,
                            title = f"Epoch [{epoch}/50] | {snapshot['id']} | ({snapshot_group})",
                        )
                    
                        
                        snapshot_data[snapshot_group][snapshot['id']].append(
                            {
                                'probability': self.fig_to_ndarray(probability_heatmap),
                                'image_mask_pred' : self.fig_to_ndarray(image_mask_pred),
                                'epoch': epoch
                            }
                        )


                        
            for snapshot_group, snapshot_group_values in snapshot_data.items():

                for snapshot_id, snapshots in snapshot_group_values.items():

                    sorted_snapshots = sorted(snapshots, key=lambda x: x['epoch'])
                    
                    probability_frames = np.array([x["probability"] for x in sorted_snapshots])
                    image_mask_pred_frames = np.array([x["image_mask_pred"] for x in sorted_snapshots])

                    # write probability animation 
                    write_animation(
                        probability_frames,
                        output_filename = f"{snapshot_id}_probability",
                        fps=2
                    )

                    # write image-mask-pred animation
                    write_animation(
                        image_mask_pred_frames,
                        output_filename = f"{snapshot_id}_progress",
                        fps=2
                    )
                    
 
                    

                

if __name__ == "__main__":
    c = VisualizeSnapshots(config = "./cam.config")
    c.main()
