import numpy as np 
import tensorflow as tf

from config import *
from utils import *
from model import *


class BenchmarkDeepLab(Utils):


    def __init__(self, config):

        self.config = config


    def main(self):

        """
        --- DATASETS ---
        """
        
        # read benchmark datasets
        benchmark_dataset = self.read_dataset(
            self.config["dataset"]["file"],
            role = "benchmark"
        )

        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops = tf.distribute.ReductionToOneDevice()
        )
        
        with strategy.scope():

            # get datapoints
            benchmark_datapoints = benchmark_dataset.datapoints


            # define model 
            model = DeeplabV3Plus(512)

            # define metrics
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name = "accuracy"),
                tf.keras.metrics.Precision(name = "precision"),
                tf.keras.metrics.Recall(name = "recall"),
                tf.keras.metrics.AUC(name = "auc"),
                tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], name = "iou")
            ]


            # load weights 
            path_to_weights = pathlib.Path(
                "./weights/", f"{self.config['deeplab']['name']}_best_weights.h5"
            ).as_posix()

            model.load_weights(path_to_weights)

            
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

            # create benchmark data
            benchmark_data = [
                {"id":datapoint_id, "metrics":datapoint_metrics} for datapoint_id, datapoint_metrics in zip([x.id for x in benchmark_datapoints], benchmark_performance)
            ]

            columns = ["id", "loss", "accuracy", "precision", "recall", "roc-auc", "iou"]
            
                              
            self.write_benchmark_results(
                "benchmark_results.csv",
                benchmark_data,
                columns = columns
            )


