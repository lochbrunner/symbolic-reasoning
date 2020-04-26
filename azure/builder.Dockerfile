FROM ubuntu:19.10

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libjpeg-dev \
    libpng-dev \
    strace && \
    rm -rf /var/lib/apt/lists/*


COPY iconv-0.2.1-cp37-cp37m-linux_x86_64.whl /tmp/
ENV PATH /opt/conda/envs/pytorch-py37/bin:/opt/conda/bin/:$PATH

RUN curl -o ~/miniconda.sh -L -O  https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install conda-build -y && \
    conda create -y --name pytorch-py37 python=3.7.5 numpy pyyaml && \
    conda clean -ya

RUN cd /tmp/ &&  \
    curl -L -O https://download.pytorch.org/whl/cpu/torch-1.3.0%2Bcpu-cp37-cp37m-linux_x86_64.whl -o pytorch.whl && \
    PATH=/opt/conda/envs/pytorch-py37/bin:$PATH pip install *.whl azureml-core && \
    rm *.whl
