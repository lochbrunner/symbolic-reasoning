#!/usr/bin/env bash

set +x

VERSION=6

TRACE_FILE=/tmp/strace.txt

mkdir -p workspace
docker run -w /workspace -v $PWD/..:/workspace -v $(dirname $TRACE_FILE):/tmp/ --entrypoint strace symbolicreasd05db995.azurecr.io/train:$VERSION-builder \
 -e trace=open,openat -f -o /tmp/$(basename $TRACE_FILE) \
 ./ml/train.py -c real_world_problems/basics/dataset.yaml -v --data-size-limit 100 -n 1 > /dev/null

./extract_used_files.py /tmp/strace.txt -o workspace/filenames.txt
# docker run -w /workspace -v $PWD/:/workspace --entrypoint python symbolicreasd05db995.azurecr.io/train:$VERSION-builder \
#     ./resolve_symlinks.py ./workspace/filenames.txt -o ./workspace/addional_filenames.txt

docker run -w /workspace -v $PWD/workspace/:/workspace/ --entrypoint bash symbolicreasd05db995.azurecr.io/train:$VERSION-builder \
    -c 'tar -C / -Pcvhf /workspace/files.tar $(cat /workspace/filenames.txt)'


docker build -f runner.Dockerfile -t symbolicreasd05db995.azurecr.io/train:$VERSION .

# Test it
docker run -w /workspace -v $PWD/..:/workspace --entrypoint /opt/conda/envs/pytorch-py37/bin/python symbolicreasd05db995.azurecr.io/train:$VERSION \
    ./ml/train.py -c real_world_problems/basics/dataset.yaml -v --data-size-limit 100 -n 1