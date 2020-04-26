# Alpine with bash
FROM amd64/alpine:bash

ENV PYTHONHOME=/opt/conda/envs/pytorch-py37
ADD workspace/files.tar /