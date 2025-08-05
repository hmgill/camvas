import cv2
import pathlib
import tqdm

import numpy as np

from skimage.morphology import skeletonize, square, dilation
from PVBM.GeometricalAnalysis import GeometricalVBMs
from PVBM.FractalAnalysis import MultifractalVBMs

# project imports 
from ...paths import project_paths
from ..data.io import DataIO
from ..processing.image_ops import ImageOperations
from ..helpers.utils import Utils



class Descriptors:

    def __init__(self, config):
        
        self.config = config

        # initialize
        self.data_io = DataIO()
        self.image_ops = ImageOperations()
        
        # read uuid - source map
        self.uuid_to_source = self.read_dill(
            pathlib.Path(
                project_paths["predictions"],
                "uuid_source_info.dill"
            ).as_posix()
        )
        
        # initialize geometric vascular biomarkers (VBMs) 
        self.geometrical_vbms = GeometricalVBMs()

        # initialize multifractal VBMs
        self.multifractal_vbms = MultifractalVBMs(
            n_rotations = 25,
            optimize = True,
            min_proba = 0.0001,
            maxproba = 0.9999
        )


    def read_dill(self, *args, **kwargs):
        return self.data_io.read_dill(*args, **kwargs)

    def get_binary_prediction_info(self, *args, **kwargs):
        return self.data_io.get_binary_prediction_info(*args, **kwargs)

    def to_range_0_1(self, *args, **kwargs):
        return self.image_ops.to_range_0_1(*args, **kwargs)
    
    def write_descriptor_file(self, *args, **kwargs):
        return self.data_io.write_descriptor_file(*args, **kwargs)

    
    def main(self):
        """
        
        --- RETRIEVE PREDICTIONS ---

        """

        # initialize_descriptors
        descriptors = list()

        # get prediction paths 
        binary_prediction_info = self.get_binary_prediction_info()

        for info in binary_prediction_info:

            # id 
            id = info["id"]
            
            # get uuid
            uuid = info["uuid"]

            # get path
            path = info["path"].as_posix()
            
            # get source
            source = self.uuid_to_source[uuid]

            """
            
            --- COMPUTE VASCULAR DESCRIPTORS ---

            """

            # read the segmentation prediction (in pixel value range [0 - 255])
            prediction_0_255 = cv2.imread(path, 0)
            
            # also read in pixel value range [0 - 1]
            prediction_0_1 = self.to_range_0_1(prediction_0_255)
            
            
            # DESCRIPTOR || area
            area = self.geometrical_vbms.area(prediction_0_1)

            # DESCRIPTOR || perimeter
            perimeter, _ = self.geometrical_vbms.compute_perimeter(prediction_0_1)

            
            # DESCRIPTOR || vascular density
            _, thresh = cv2.threshold(
                prediction_0_255,
                250,
                255,
                cv2.THRESH_BINARY_INV
            )
            vessel_density = (np.sum(prediction_0_1 == np.max(prediction_0_1)) / np.sum(thresh == np.max(thresh))*100)

            # skeletonize
            skeleton = skeletonize(prediction_0_1).astype(np.uint8)

            
            # DESCRIPTOR || tortuosity, total length, chord/arc, dico
            median_tortuosity, total_length, chord, arc, connection_dico = self.geometrical_vbms.compute_tortuosity_length(
                skeleton
            )
            # sanity check 
            assert len(chord) == len(arc)
        
            # DESCRIPTOR || number of vessels
            num_vessels = len(chord)

        
            # DESCRIPTOR || average length
            average_length = (total_length / num_vessels)

            
            # DESCRIPTOR || number of end points and branch points
            num_end_points, num_branch_points, end_points, branch_points = self.geometrical_vbms.compute_particular_points(skeleton)
            
            # DESCRIPTOR || end point density
            end_point_density = ((num_end_points / np.sum(thresh == 255)) * 100)

            # DESCRIPTOR || branch point density
            branch_point_density = ((num_branch_points / np.sum(thresh == 255)) * 100)

            # DESCRIPTOR || mean branch angle, STD branch angle, median branch angle 
            mean_branch_angle, std_branch_angle, median_branch_angle, angle_dico, centroid = self.geometrical_vbms.compute_branching_angles(skeleton)

            # DESCRIPTOR || fractal capacity, entropy, oorrelation dimensions, singularity length 
            f_dim_capacity, f_dim_entropy, f_dim_corr, f_sing = self.multifractal_vbms.compute_multifractals(prediction_0_1.astype(np.float64))

            # organize all of the computed vascular descriptors into a dict
            prediction_descriptors = {
                "id" : id,
                "source" : source,
                "num_vessels" : num_vessels,
                "vessel_area" : area,
                "vessel_perimeter" : perimeter,
                "vessel_density" : vessel_density,
                "vessel_total_length" : total_length,
                "vessel_average_length" : average_length,
                "vessel_median_tortuosity" : median_tortuosity,
                "num_branch_points" : num_branch_points,
                "num_end_points" : num_end_points,
                "branch_point_density" : branch_point_density,
                "end_point_density" : end_point_density,
                "mean_vessel_branch_angle" : mean_branch_angle,
                "median_vessel_branch_angle" : median_branch_angle,
                "std_vessel_branch_angle" : std_branch_angle,
                "vessel_fractal_capacity_dimension" : f_dim_capacity,
                "vessel_fractal_entropy_dimension" : f_dim_entropy,
                "vessel_fractal_correlation_dimension" : f_dim_corr,
                "vessel_fractal_singularity_length" : f_sing            
            }

            # round to suitable decimal places
            round_to_1 = ["vessel_area", "vessel_perimeter", "vessel_area", "vessel_density", "vessel_average_length", "branch_point_density", "end_point_density", "mean_vessel_branch_angle", "median_vessel_branch_angle", "std_vessel_branch_angle"]
            round_to_3 = ["vessel_median_tortuosity", "vessel_fractal_capacity_dimension", "vessel_fractal_entropy_dimension", "vessel_fractal_correlation_dimension", "vessel_fractal_singularity_length"]
            for k, v in prediction_descriptors.items():
                if k in round_to_1:
                    prediction_descriptors[k] = np.round(v, 1)
                elif k in round_to_3:
                    prediction_descriptors[k] = np.round(v, 3)
            
            # aggregate
            descriptors.append(prediction_descriptors)


            
        # write dataset descriptors results to CSV file
        self.write_descriptor_file(descriptors)
        
