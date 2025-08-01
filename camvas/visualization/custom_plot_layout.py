import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np

def create_split_legend(ax, legend_data, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15)):
    """
    Create a legend with diagonally split color boxes.
    
    Parameters:
    - ax: matplotlib axis
    - legend_data: list of dicts with 'metric', 'train_hex', 'val_hex'
    - loc: legend location
    - ncol: number of columns (items per row)
    - bbox_to_anchor: position of legend relative to axes
    """
    
    # Create custom legend handles with labels stored
    legend_handles = []
    legend_labels = []
    
    for i, item in enumerate(legend_data):
        # Create a custom patch with a label attribute
        handle = mpatches.Rectangle((0, 0), 1, 1, 
                                   facecolor='none',
                                   edgecolor='none',
                                   linewidth=0,
                                   label=item['metric'])  # Store label in handle
        legend_handles.append(handle)
        legend_labels.append(item['metric'])
    
    # Create the legend with no background
    legend = ax.legend(legend_handles, legend_labels, 
                      loc=loc, ncol=ncol,
                      bbox_to_anchor=bbox_to_anchor,
                      frameon=False,  # No background
                      handler_map={mpatches.Rectangle: DiagonalSplitHandler(legend_data)})
    
    return legend


class DiagonalSplitHandler:
    """Custom legend handler for diagonally split color boxes."""
    
    def __init__(self, legend_data):
        self.legend_data = legend_data
        # Create a mapping from metric names to data
        self.data_map = {item['metric']: item for item in legend_data}
        
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        # Get the label from the handle
        label = orig_handle.get_label()
        
        # Look up the data for this metric
        if label in self.data_map:
            item = self.data_map[label]
            
            # Create two triangular polygons split diagonally
            # Left triangle (train) - bottom-left, top-left, bottom-right
            train_triangle = Polygon([(x0, y0), 
                                    (x0, y0 + height), 
                                    (x0 + width, y0)],
                                   facecolor=item['train_hex'],
                                   edgecolor='black',
                                   linewidth=0.5,
                                   transform=handlebox.get_transform())
            
            # Right triangle (val) - top-left, top-right, bottom-right
            val_triangle = Polygon([(x0, y0 + height), 
                                  (x0 + width, y0 + height), 
                                  (x0 + width, y0)],
                                 facecolor=item['val_hex'],
                                 edgecolor='black',
                                 linewidth=0.5,
                                 transform=handlebox.get_transform())
            
            handlebox.add_artist(train_triangle)
            handlebox.add_artist(val_triangle)
            
        return None
