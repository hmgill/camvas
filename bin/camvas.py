
import argparse

from core.config import *
from core.train_deeplab import * 
from core.benchmark_deeplab import *
from core.predict_deeplab import *
from core.compute_vascular_descriptors import * 


def main(args):

    # get config 
    config = ConfigReader().read_config(args.config)
    
    # get mode 
    mode = args.mode


    
    # train
    if mode == "train":
        TrainDeepLab(config).main()        

    # benchmark
    elif mode == "benchmark":
        BenchmarkDeepLab(config = config).main()

    # predict 
    elif mode == "predict":
        PredictDeepLab(config).main()

    # compute vascular descriptors 
    elif mode == "descriptors":
        Descriptors(config).main()
    
    
        
    

    





if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='')

   # Add required arguments
   parser.add_argument('--config',
                       required=True,
                       help='Path to the configuration file')
   parser.add_argument('--mode',
                       required=True,
                       choices=['train', 'benchmark', 'predict', 'descriptors'],
                       help='Operation mode: train, benchmark, predict, or descriptors')


   
   # Add report argument group
   report_group = parser.add_argument_group('report', 'Report generation options')
   report_group.add_argument('--report',
                             action='store_true', help='Generate a report')
   
   # Parse known args first to check if report is enabled
   args, remaining = parser.parse_known_args()

   # Add conditional report parameters if report is True
   if args.report:
       report_group.add_argument('--report-format',
                                 choices=["html"],
                                 default="html",
                                 help='report format')
       report_group.add_argument('--report-output',
                                 help='output path for report')
       
       # Parse all arguments again with the conditional ones added
       args = parser.parse_args()

       
   else:
       # Parse remaining arguments normally
       args = parser.parse_args()

    
   main(args)
   

