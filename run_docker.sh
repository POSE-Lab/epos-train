BOP_PATH=$1
STORE_PATH=$2
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $BOP_PATH:/home/datasets -v $STORE_PATH:/home/store -v $(pwd)/scripts:/home/epos/scripts -v $(pwd)/epos_lib:/home/epos/epos_lib -v $(pwd)/docker/dataset_params_industry.py:/home/epos/external/bop_toolkit/bop_toolkit_lib/dataset_params.py epos-train-nv2303:latest bash