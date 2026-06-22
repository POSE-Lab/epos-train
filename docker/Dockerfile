ARG IMAGE_REPO=ghcr.io/athnzc/
FROM ${IMAGE_REPO}ubuntu2004_cuda10:latest AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    llvm \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev 

COPY ./docker/packages.txt ./packages.txt
RUN apt-get update && xargs -a packages.txt apt-get install -y
RUN apt-get update && apt-get install -y python3-dev python-is-python3
WORKDIR /tmp
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


RUN conda create -y -n epos python=3.6.10 pip

RUN conda install -y -n epos -c conda-forge \
    pandas=1.0.5 scipy=1.5.3 pyyaml=5.3.1

RUN conda install -y -n epos \
    imageio=2.9.0 tabulate=0.8.3 
    
RUN conda install -y -n epos -c mjirik pypng=0.0.18

RUN conda install -y -n epos \
    gflags

RUN conda install -y -n epos \
    opencv=3.4.2

RUN conda install -y -n epos -c conda-forge glog=0.4.0
# RUN conda install -y -n epos -c conda-forge \
#     igl=2.2.1

 SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate epos && \
    pip install \
      libigl \
      tensorflow-gpu==1.14.0 \
      tensorboard==1.14.0 \
      tensorflow-estimator==1.14.0 \
      protobuf==3.19.6 \
      h5py==3.1.0

ENV REPO_PATH=/home/epos
ENV CONDA_PREFIX=/opt/conda/envs/epos
WORKDIR $REPO_PATH
COPY ./docker/env_vars.sh $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
COPY ./external ./external

# The installation locations of OSMesa and LLVM:
ENV OSMESA_PREFIX=$REPO_PATH/external/osmesa
ENV LLVM_PREFIX=$REPO_PATH/external/llvm

COPY ./docker/mesa/ .
RUN source /opt/conda/etc/profile.d/conda.sh && \
conda activate epos && mkdir -p $OSMESA_PREFIX && mkdir -p $LLVM_PREFIX && cd $REPO_PATH/external/bop_renderer/osmesa-install && \
mkdir build; cd build && \
cp $REPO_PATH/osmesa-install.sh $REPO_PATH/external/bop_renderer/osmesa-install/ && \
cp $REPO_PATH/glu-9.0.0.tar.bz2 $REPO_PATH/external/bop_renderer/osmesa-install/build && \
bash ../osmesa-install.sh

RUN source /opt/conda/etc/profile.d/conda.sh && \
conda activate epos && export PYTHON_PREFIX=$CONDA_PREFIX && mkdir -p $REPO_PATH/external/bop_renderer/build && \
cd $REPO_PATH/external/bop_renderer/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8

RUN source /opt/conda/etc/profile.d/conda.sh && \
conda activate epos && mkdir -p external/progressive-x/build && \
    cd external/progressive-x/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF &&\
    make -j$(nproc)

RUN source /opt/conda/etc/profile.d/conda.sh && \
conda activate epos && pip uninstall -y gast && pip install gast==0.2.2