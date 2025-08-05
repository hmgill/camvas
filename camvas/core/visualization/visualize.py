import pathlib 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from colorutils import Color

from ...paths import project_paths
from .animate import * 



class Visualize:
    
    
    def make_probability_heatmap(self, image_array, prediction_array, title="Pixel Value Heatmap", figsize=(16, 8),
                                 colormap='viridis', show_colorbar=True, interpolation='bilinear'):
        
        """
        Visualize a 2D array as a heatmap with color mapping from min to max values.

        Parameters:
        -----------
        image_array : numpy.ndarray
        Input image array with shape (512, 512, 1) or (512, 512)
        prediction_array : numpy.ndarray, optional
        Prediction array with same shape as image_array. If None, only shows image.
        title : str
        Title for the plot
        figsize : tuple
        Figure size as (width, height) - default (16, 8) for two subplots
        colormap : str or matplotlib colormap
        Colormap to use. Options: 'viridis', 'plasma', 'inferno', 'magma',
        'coolwarm', 'hot', 'jet', or custom colormap
        show_colorbar : bool
        Whether to show the colorbar
        interpolation : str
        Interpolation method for imshow
        save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        dpi : int
        DPI for saved figure (default: 300, higher values may cause memory issues)
        close_fig : bool
        Whether to close the figure after saving to free memory (default: True)

        Returns:
        --------
        fig : matplotlib figure 
        """


        # Handle prediction array if provided
        if prediction_array is not None:
            if prediction_array.ndim == 3 and prediction_array.shape[2] == 1:
                pred_data = prediction_array.squeeze()
            elif prediction_array.ndim == 2:
                pred_data = prediction_array
            else:
                raise ValueError(f"Expected prediction_array with shape (512, 512, 1) or (512, 512), got {prediction_array.shape}")

        # Two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]

        # Get min and max values for consistent scaling
        vmin, vmax = np.min(pred_data), np.max(pred_data)

        # Create the heatmaps
        im1 = ax1.imshow(image_array, aspect='equal')
        im2 = ax2.imshow(pred_data, cmap=colormap, vmin=vmin, vmax=1,
                         interpolation=interpolation, aspect='equal')

        # Set titles and remove axis ticks/labels
        ax1.set_title('Image', fontsize=12)
        ax2.set_title('Pixel Probabilities', fontsize=12)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # Add colorbar at the bottom spanning both subplots
        if show_colorbar:
            plt.subplots_adjust(bottom=0.15)

            # Create colorbar that spans both subplots at the very bottom
            cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])  # [left, bottom, width, height]
            cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=10)

        # Add value range info to upper left corner of ax2
        ax2.text(0.02, 0.98, f'Range: [0, {str(np.round(vmax, 2))}]',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set main title
        fig.suptitle(title, fontsize=14, y=0.95)

        return fig
   


    
    def make_comparison_fig(self, image_array, prediction_array, overlay, title="", figsize=(24, 8),
                                 colormap='viridis', show_colorbar=True, interpolation='bilinear'):
        """
        Visualize a 2D array as a heatmap with color mapping from min to max values.

        Parameters:
        -----------
        image_array : numpy.ndarray
        Input image array with shape (512, 512, 1) or (512, 512)
        prediction_array : numpy.ndarray, optional
        Prediction array with same shape as image_array. If None, only shows image.
        title : str
        Title for the plot
        figsize : tuple
        Figure size as (width, height) - default (16, 8) for two subplots
        colormap : str or matplotlib colormap
        Colormap to use. Options: 'viridis', 'plasma', 'inferno', 'magma',
        'coolwarm', 'hot', 'jet', or custom colormap
        show_colorbar : bool
        Whether to show the colorbar
        interpolation : str
        Interpolation method for imshow
        save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        dpi : int
        DPI for saved figure (default: 300, higher values may cause memory issues)
        close_fig : bool
        Whether to close the figure after saving to free memory (default: True)

        Returns:
        --------
        fig, (ax1, ax2) : matplotlib figure and axis objects (or None if closed)
        """


        # Handle prediction array if provided
        if prediction_array is not None:
            if prediction_array.ndim == 3 and prediction_array.shape[2] == 1:
                pred_data = prediction_array.squeeze()
            elif prediction_array.ndim == 2:
                pred_data = prediction_array
            else:
                raise ValueError(f"Expected prediction_array with shape (512, 512, 1) or (512, 512), got {prediction_array.shape}")

        # Two subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        axes = [ax1, ax2, ax3]

        # Get min and max values for consistent scaling
        vmin, vmax = np.min(pred_data), np.max(pred_data)

        # Create the heatmaps
        im1 = ax1.imshow(image_array[:,:,::-1], aspect='equal')
        im2 = ax2.imshow(pred_data, cmap=colormap, vmin=vmin, vmax=1,
                         interpolation=interpolation, aspect='equal')
        im3 = ax3.imshow(overlay[:,:,::-1], aspect='equal')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])


        # Add value range info to upper left corner of ax2
        ax2.text(0.02, 0.98, f'Range: [0, {str(np.round(vmax, 2))}]',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


        return fig 





'''
class VisualizeSnapshots(Utils):


    def __init__(self, config, snapshot_dill = project_paths["snapshot_dill"], snapshot_plots = project_paths["snapshot_plots"]):

        self.config = config
        self.snapshot_dill = snapshot_dill
        self.snapshot_plots = snapshot_plots
        
    

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
            pathlib.Path(self.snapshot_dill).glob("*.snapshots.dill")
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
'''


 
                    

                
