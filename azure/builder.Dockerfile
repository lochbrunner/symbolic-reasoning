FROM ubuntu:20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libjpeg-dev \
    libpng-dev \
    strace \
    file \
    && rm -rf /var/lib/apt/lists/*


COPY tmp/iconv-0.3.3-cp38-cp38-linux_x86_64.whl /tmp/
# COPY tmp/torch-1.7.0+cu110-cp38-cp38-linux_x86_64.whl /tmp/
ENV PATH /opt/conda/envs/pytorch-py37/bin:/opt/conda/bin/:$PATH

RUN curl -o ~/miniconda.sh -L -O  https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install conda-build -y && \
    conda create -y --name pytorch-py38 python=3.8.5 numpy pyyaml py-cpuinfo && \
    conda clean -ya

# PyTorch Wheels can be found here: https://download.pytorch.org/whl/torch_stable.html
RUN cd /tmp/ &&  \
    curl -L -O https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl -o pytorch.whl && \
    PATH=/opt/conda/envs/pytorch-py37/bin:$PATH pip install *.whl tensorboard matplotlib azureml-core azureml-tensorboard && \
    rm *.whl
