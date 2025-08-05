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



class BenchmarkDeepLab:


    def __init__(self, config):

        self.config = config

        # initialize
        self.data_io = DataIO()
        self.image_ops = ImageOperations()
        self.utils = Utils(config)


    def read_dataset(self, *args, **kwargs):
        """Delegate to data_io component."""
        return self.data_io.read_dataset(*args, **kwargs)

    def get_ids_from_dataset(self, *args, **kwargs):
        """Delegate to data_io component."""
        return self.utils.get_ids_from_dataset(*args, **kwargs)

    def create_tf_dataset(self, *args, **kwargs):
        """Delegate to data_io component."""
        return self.utils.create_tf_dataset(*args, **kwargs)

    def get_best_weights(self, *args, **kwargs):
        return self.data_io.get_best_weights(*args, **kwargs)
    
    def write_dill(self, *args, **kwargs):
        """Delegate to data_io component."""
        return self.data_io.write_dill(*args, **kwargs)


    def write_benchmark_results(self, *args, **kwargs):
        """ Write Benchmark Results to CAMvas output directory """
        return self.data_io.write_benchmark_results(*args, **kwargs)
    

    def main(self):

        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops = tf.distribute.ReductionToOneDevice()
        )
        
        with strategy.scope():
            
            """
            --- DATASETS ---
            """
            
            # read benchmark datasets
            benchmark_dataset = self.read_dataset(
                self.config["dataset"]["file"],
                role = "benchmark"
            )
            
            # get datapoints
            benchmark_datapoints = benchmark_dataset.datapoints

            # define model 
            model = DeepLabV3Plus()

            # load weights
            path_to_weights = self.get_best_weights(self.config['deeplab']['name'])
            model.load_weights(path_to_weights)

            # define metrics
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name = "accuracy"),
                tf.keras.metrics.Precision(name = "precision"),
                tf.keras.metrics.Recall(name = "recall"),
                tf.keras.metrics.AUC(name = "auc"),
                tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], name = "iou")
            ]
            
            # compile model
            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = self.config["training"]["learning_rate"]),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = metrics
            )

            benchmark_performance = []
            for datapoint in benchmark_datapoints:
                # create data generators for benchmark / external data
                benchmark_generator = self.create_tf_dataset(
                    np.array([datapoint])
                )
                # evaluate performance on benchmark CAM datasets
                benchmark_performance.append(
                    model.evaluate(benchmark_generator)
                )

            # create benchmark data and write results
            benchmark_data = [
                {"id":datapoint_id, "metrics":datapoint_metrics} for datapoint_id, datapoint_metrics in zip([x.id for x in benchmark_datapoints], benchmark_performance)
            ]

            columns = ["id", "loss", "accuracy", "precision", "recall", "roc-auc", "iou"]
            
            self.write_benchmark_results(
                benchmark_data,
                columns = columns
            )
