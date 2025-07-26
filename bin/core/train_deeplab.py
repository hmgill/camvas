import sys
import pathlib

parent_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np 
import tensorflow as tf

from sklearn.model_selection import KFold


from model.network import *
from model.custom_metrics import * 
from config import *
from utils import *
from callbacks import * 


class TrainDeepLab(Utils):


    def __init__(self, config):

        self.config = config


    def main(self):

        """
        --- DATASETS ---
            
        train_dataset : 
            
        benchmark_dataset :

        """
        
        # read model training dataset
        train_dataset = self.read_dataset(
            self.config["dataset"]["file"],
            role = "train"
        )

        # read benchmark datasets
        benchmark_dataset = self.read_dataset(
            self.config["dataset"]["file"],
            role = "benchmark"
        )

        # read in-ovo CAM dataset
        in_ovo_dataset = self.read_dataset(
            self.config["dataset"]["file"],
            role = "benchmark",
            include_source = "in-ovo-cam"
        )

        # read ex-ovo CAM dataset
        ex_ovo_dataset = self.read_dataset(
            self.config["dataset"]["file"],
            role = "benchmark",
            include_source = "ex-ovo-cam"
        )


        """
        --- NESTED K-FOLD CROSS VALIDATION ---
        """

        outer_kf = KFold(n_splits = self.config["kfold"]["k"])
        inner_kf = KFold(n_splits = self.config["kfold"]["k"])

        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops = tf.distribute.ReductionToOneDevice()
        )
        
        with strategy.scope():

            # initialize best iou score
            best_score = 0 
            
            # convert datapoints to numpy arrays
            train_datapoints = train_dataset.datapoints
            benchmark_datapoints = benchmark_dataset.datapoints
            in_ovo_datapoints = in_ovo_dataset.datapoints
            ex_ovo_datapoints = ex_ovo_dataset.datapoints

            # create data generators for benchmark / external data
            benchmark_generator = self.create_tf_dataset(
                benchmark_datapoints
            )
            in_ovo_generator =  self.create_tf_dataset(
                in_ovo_datapoints
            )
            ex_ovo_generator =  self.create_tf_dataset(
                ex_ovo_datapoints
            )
            
            
            """
            outer folds
            """

            for outer_fold, (inner_fold_idx, outer_fold_idx) in enumerate(outer_kf.split(train_datapoints)):

                # create internal test dataset for outer fold
                test_data = train_datapoints[outer_fold_idx]
                test_generator = self.create_tf_dataset(
                    test_data
                )

                # remaining data for inner folds
                inner_data = train_datapoints[inner_fold_idx]

                """
                innner folds
                """
                
                for inner_fold, (train_idx, val_idx) in enumerate(inner_kf.split(inner_data)):

                    # set ID for nested k-fold cross-validation fold
                    fold_id = f"{str(outer_fold)}_{str(inner_fold)}"
                    

                    # set train and val datasets
                    train_data = inner_data[train_idx]
                    val_data = inner_data[val_idx]

                    train_generator = self.create_tf_dataset(
                        train_data,
                        apply_augmentation = True
                    )

                    val_generator = self.create_tf_dataset(
                        val_data
                    )

                    
                    # get fold data information 
                    fold_data = {
                            "train" : train_data,
                            "val" : val_data,
                            "test" : test_data,
                            "in-ovo-cam" : in_ovo_datapoints,
                            "ex-ovo-cam" : ex_ovo_datapoints
                        }

                    fold_data_ids = self.get_ids_from_dataset(
                        fold_data
                    )
                                        

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

                    # compile model 
                    model.compile(
                        optimizer = tf.keras.optimizers.Adam(learning_rate = self.config["training"]["learning_rate"]),
                        loss = tf.keras.losses.BinaryCrossentropy(),
                        metrics = metrics
                    )
                    
                    # define callbacks
                    callbacks = [
                        Snapshots(
                            fold_id,
                            fold_data
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_iou',
                            mode='max',
                            factor=0.5,
                            patience=15,
                            min_lr=1e-6,
                            verbose=1
                        )
                    ]

                    # fit model 
                    history = model.fit(
                        train_generator,
                        validation_data = val_generator,
                        epochs = self.config["training"]["epochs"],
                        callbacks = callbacks
                    )

                    # evaluate model performance on test data
                    test_performance = model.evaluate(test_generator)

                    # evaluate performance on benchmark CAM datasets
                    benchmark_performance = model.evaluate(benchmark_generator)
                    in_ovo_performance = model.evaluate(in_ovo_generator)
                    ex_ovo_performance = model.evaluate(ex_ovo_generator)

                    # update best weights based on benchmark dataset IoU score 
                    benchmark_iou = benchmark_performance[-1]

                    if benchmark_iou >= best_score:
                   
                        best_score = benchmark_iou

                        output_path = pathlib.Path(
                            "./weights/", f"{self.config['deeplab']['name']}_best_weights.h5"
                        ).as_posix()
            
                        model.save_weights(output_path)

                
                    # save history information
                    performance_dict = {
                        "train_val" : history.history,
                        "test" : test_performance,
                        "benchmark" : benchmark_performance,
                        "in-ovo" : in_ovo_performance,
                        "ex-ovo" : ex_ovo_performance
                    }

                    history_dict = {
                        "id" : fold_id,
                        "performance" : performance_dict,
                        "data_ids" : fold_data_ids
                    }

                    self.write_dill(
                        f"{fold_id}.history",
                        history_dict
                    )
                    
