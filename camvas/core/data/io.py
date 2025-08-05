import cv2
import dill
import pathlib
import shortuuid
import tqdm

import numpy as np 
import pandas as pd 

from loguru import logger
from typing import List, Dict, Any, Optional, Union

from ...paths import project_paths
from ..helpers.decorators import check_output_dir
from .classes import Dataset, Datapoint, ImageMaskOrPrediction



class DataIO():


    def __init__(self):
        pass
    

    def read_dataset(self,
                     dataset_file,
                     role = "train",
                     include_source = None,
                     exclude_source = None,
                     resize = 512) -> Dataset:

        # read dataset CSV file
        df = pd.read_csv(dataset_file)

        # filter DF by role
        df = df[df['role'] == role]

        # filter by name (if given)
        if include_source:
            df = df[df['source'] == include_source]

        # filter to exclude name (if given)
        if exclude_source:
            df = df[df['source'] != exclude_source]

        # Convert NA values to None
        df = df.where(pd.notna(df), None)

        # convert df into list of dicts
        dataset_items = df.to_dict('records')

        # iterate over
        datapoints = list()

        for dataset_item in tqdm.tqdm(dataset_items):

            image = ImageMaskOrPrediction(
                id = dataset_item["id"],
                path = dataset_item["image"],
                shape = (resize, resize, 3)
            )

            mask = ImageMaskOrPrediction(
                id = dataset_item["id"],
                path = dataset_item["mask"],
                shape = (resize, resize, 1)
            )

            datapoint = Datapoint(
                id = dataset_item["id"],
                uuid = shortuuid.uuid(),
                source = dataset_item["source"],
                image = image,
                mask = mask
            )

            datapoints.append(datapoint)

        datapoints = np.array(datapoints)

        dataset = Dataset(
            datapoints = datapoints,
            role = role
        )

        return dataset


    
    @check_output_dir
    def write_dill(self, filename, output_dir, data):

        # merge output dir and filename
        output_path = pathlib.Path(output_dir, filename)

        # add .dill extension if not given
        if output_path.suffix != ".dill":
            output_path = pathlib.Path(output_dir, f"{filename}.dill")

        # write dill file
        with open(output_path.as_posix(), 'wb') as f:
            dill.dump(data, f)



    def read_dill(self, filename):
        with open(filename, 'rb') as f:
            data = dill.load(f)
        return data



    def file_is_csv(self, file_path):
        path_obj = pathlib.Path(file_path)
        return path_obj.suffix == '.csv'



    @check_output_dir
    def write_benchmark_results(self, data, columns = None, filename : str = "benchmark.csv", output_dir = project_paths["output"]):

        # merge output dir and filename
        output_path = pathlib.Path(output_dir, filename)

        # insert ID to first index of datapoint lists
        for idx, datapoint in enumerate(data):
            datapoint["metrics"].insert(0, datapoint["id"])
            data[idx] = datapoint["metrics"]

        # create DF from data
        df = pd.DataFrame(data, columns = columns)

        # write DF to CSV
        if not self.file_is_csv(output_path):
            output_path.with_suffix(".csv")

        df.to_csv(
            output_path.as_posix(),
            index = False
        )


    def get_best_weights(self, model_name):
        return pathlib.Path(
            project_paths["weights"], f"{model_name}_best_weights.h5"
        ).as_posix()
        


    @check_output_dir
    def write_prediction_result(self, data, output_dir = project_paths["predictions"]):

        # create unique output directory for prediction results
        prediction_output_dir = pathlib.Path(
            output_dir,
            f"{data['id']}_{data['uuid']}"
        )

        prediction_output_dir.mkdir()

        # write prediction plots
        for plot in data["plots"]:

            plot_path = pathlib.Path(
                prediction_output_dir,
                plot["filename"]
            ).as_posix()

            plot_data = plot["data"]

            cv2.imwrite(
                plot_path,
                plot_data
            )

        

    
    @check_output_dir
    def write_descriptor_file(self, data, filename="descriptors.csv", output_dir = project_paths["output"]):

        # merge output dir and filename
        output_path = pathlib.Path(output_dir, filename)

        # write as CSV 
        df = pd.DataFrame.from_dict(data)
        df = df.set_index('id')
        df.to_csv(output_path.as_posix())
        

    def get_binary_prediction_info(self, prediction_dir = project_paths["predictions"]):

        dirs = [x for x in list(prediction_dir.glob("*")) if x.is_dir()]
        info = [{"id":"_".join(x.stem.split("_")[:-1]), "uuid":x.stem.split("_")[-1], "path":pathlib.Path(x, "binary_prediction.png")} for x in dirs]

        return info 
        

