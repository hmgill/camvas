#!/bin/bash
# Build CAMVAS Apptainer container without sudo

module load apptainer 

# Set build options
export APPTAINER_TMPDIR=/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=/tmp/apptainer_cache

# Create temporary directories
mkdir -p $APPTAINER_TMPDIR
mkdir -p $APPTAINER_CACHEDIR

echo "Building CAMVAS Apptainer container..."
echo "This may take several minutes..."

apptainer build --fakeroot camvas.sif camvas.def

echo "Build complete!"
echo "Container saved as: camvas.sif"

