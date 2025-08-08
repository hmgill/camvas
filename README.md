# CAMVAS
<div align="center">
<img src="media/camvas-cartoon.png" alt="logo" width="60%">
</div>

## Setup

### Apptainer 

To create an Apptainer image of CAMVAS, run:

```./build_camvas.sh```

camvas.sif can then be used 

```apptainer run --bind .my_output/:/opt/output camvas.sif -m camvas.camvas --config my_config.config --mode train```

### Conda

To install CAMVAS through Conda, run: 

``` conda env create -f environment.yml ```

## Usage

## Examples
