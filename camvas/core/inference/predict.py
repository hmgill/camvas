import numpy as np 
import tensorflow as tf

# Model Imports
from ..models.deeplab import *
from ..models.metrics import *

# Other imports
from ...paths import project_paths
from ..data.io import DataIO
from ..processing.image_ops import ImageOperations
from ..helpers.utils import Utils
from ..visualization.visualize import Visualize



class PredictDeepLab:


    def __init__(self, config):

        self.config = config

        # initialize
        self.data_io = DataIO()
        self.image_ops = ImageOperations()
        self.utils = Utils(config)
        self.visualize = Visualize()


    def read_dataset(self, *args, **kwargs):
        return self.data_io.read_dataset(*args, **kwargs)

    def get_best_weights(self, *args, **kwargs):
        return self.data_io.get_best_weights(*args, **kwargs)

    def write_dill(self, *args, **kwargs):
        return self.data_io.write_dill(*args, **kwargs)

    def write_prediction_result(self, *args, **kwargs):
        return self.data_io.write_prediction_result(*args, **kwargs)

    def process(self, *args, **kwargs):
        return self.utils.process(*args, **kwargs)
    
    def fig_to_ndarray(self, *args, **kwargs):
        return self.image_ops.fig_to_ndarray(*args, **kwargs)

    def read_image_or_mask(self, *args, **kwargs):
        return self.image_ops.read_image_or_mask(*args, **kwargs)

    def make_probability_heatmap(self, *args, **kwargs):
        return self.visualize.make_probability_heatmap(*args, **kwargs)

    def make_comparison_fig(self, *args, **kwargs):
        return self.visualize.make_comparison_fig(*args, **kwargs)
        
    def threshold_prediction(self, *args, **kwargs):
        return self.utils.threshold_prediction(*args, **kwargs)
    
    def grayscale_to_color(self, *args, **kwargs):
        return self.image_ops.grayscale_to_color(*args, **kwargs)

    def overlay(self, *args, **kwargs):
        return self.image_ops.overlay(*args, **kwargs)

    def to_range_0_255(self, *args, **kwargs):
        return self.image_ops.to_range_0_255(*args, **kwargs)
    
    def save_uuid_source_info(self, datapoints):
        # create uuid:source dict 
        uuid_source_info = {x.uuid : x.source for x in datapoints}

        # save dict in predictions directory
        self.write_dill(
            "uuid_source_info.dill",
            project_paths["predictions"],            
            uuid_source_info
        )    


    def get_prediction_info(self, datapoint, model):
        # initialize
        prediction_info = {
            "id" : datapoint.id,
            "uuid" : datapoint.uuid,
            "plots" : []
        }
        # read image 
        image = self.process(datapoint, image_only = True)

        # expand image dims 
        image = np.expand_dims(image, axis = 0)

        # get the prediction  
        prediction = model.predict(image)

        # squeeze the image prediction by axis 0 to return to original image shape
        image = np.squeeze(image, axis = 0)
        prediction = np.squeeze(prediction, axis = 0)

        # normalize image to range [0 - 255]
        image = self.to_range_0_255(image)

        """
        probability heatmap 
        """
        if self.config["deeplab"]["output"]["probability_heatmap"]:

            probability_heatmap = self.fig_to_ndarray(
                self.make_probability_heatmap(
                    image,
                    prediction
                )
            )
        
            prediction_info["plots"].append(
                {
                    "filename" : "heatmap.png",
                    "data" : probability_heatmap
                }
            )

        """
        binary prediction 
        """
        binary_prediction = self.threshold_prediction(prediction.copy())

        prediction_info["plots"].append(
            {
                "filename" : "binary_prediction.png",
                "data" : binary_prediction
            }
        )

        """
        overlay
        """
        if self.config["deeplab"]["output"]["overlay"]:
                
            color_prediction = self.grayscale_to_color(binary_prediction, {255:"#148ef6"})
            overlay = self.overlay(image, color_prediction)

            prediction_info["plots"].append(
                {
                    "filename" : "overlay.png",
                    "data" : overlay
                }
            )

        """
        comparison 
        """
        if self.config["deeplab"]["output"]["comparison"]:

            comparison = self.fig_to_ndarray(
                self.make_comparison_fig(
                    image,
                    binary_prediction,
                    overlay
                )
            )
            
            prediction_info["plots"].append(
                {
                    "filename" : "comparison.png",
                    "data" : comparison
                }
            )

        return prediction_info 
            

        
        
    def main(self):

        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops = tf.distribute.ReductionToOneDevice()
        )
        
        with strategy.scope():
            
            """
            --- DATASETS ---
            """

            # read benchmark datasets
            prediction_dataset = self.read_dataset(
                self.config["dataset"]["file"],
                role = "benchmark"
            )

            # get datapoints
            prediction_datapoints = prediction_dataset.datapoints

            # save dataset uuid-source map  
            self.save_uuid_source_info(prediction_datapoints)


            """
            --- MODEL ---
            """
            
            # define model
            model = DeepLabV3Plus()

            # load weights
            path_to_weights = self.get_best_weights(self.config['deeplab']['name'])
            model.load_weights(path_to_weights)


            """
            --- PREDICTION --- 
            """

            for datapoint in prediction_datapoints:

                # get prediction info 
                prediction_info = self.get_prediction_info(
                    datapoint,
                    model
                )

                # write prediction results
                self.write_prediction_result(
                    prediction_info
                )

