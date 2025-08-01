import numpy as np 
import tensorflow as tf
import uuid

from config import *
from utils import *
from model import *


class PredictDeepLab(Utils):


    def __init__(self, config):

        self.config = config


        
    def save_uuid_source_info(self, datapoints):
        """

        """
        # create uuid:source dict 
        uuid_source_info = {x.uuid : x.source for x in datapoints}

        # save dict in predictions directory
        self.write_dill(
            "uuid_source_info.dill",
            uuid_source_info,
            output_dir = "./output/predict/"
        )    

        
    def main(self):

        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops = tf.distribute.ReductionToOneDevice()
        )
        
        with strategy.scope():
            
            """
            --- DATASETS ---
            """
            
            # read prediction dataset
            dataset = self.read_dataset(
                self.config["dataset"]["file"],
                role = "benchmark"
            )
            # save dataset uuid-source map  
            self.save_uuid_source_info(dataset.datapoints)


            """
            --- MODEL ---
            """
            
            # define model
            model = DeeplabV3Plus(512)

            # load saved model weights
            weights_path = self.get_weights_path(
                self.config['deeplab']['name']
            )
            
            model.load_weights(
                weights_path.as_posix()
            )


            """
            --- PREDICTION --- 
            """

            for datapoint in dataset.datapoints:

                self.write_prediction_result(
                    datapoint,
                    model
                )
                
