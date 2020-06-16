#!/usr/bin/env bash

set -ex

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

VERSION=6

# Creating temporary folder
tmp_dir=/tmp/$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 10 ; echo '')/
trace_file=$tmp_dir/strace.txt


mkdir -p workspace
echo -e "${GREEN}Tracing...$NC"
docker run -w /workspace -v $PWD/..:/workspace -v $(dirname $trace_file):/tmp/ --entrypoint strace symbolicreasd05db995.azurecr.io/train:$VERSION-builder \
 -e trace=open,openat -f -o /tmp/$(basename $trace_file) \
 ./ml/train.py -c real_world_problems/basics/dataset.yaml -v --data-size-limit 100 -n 1 > /dev/null

echo -e "${GREEN}Creating file list...$NC"
./extract_used_files.py /tmp/strace.txt -o workspace/filenames.txt

echo -e "${GREEN}Packing files...$NC"
docker run -w /workspace -v $PWD/:/workspace/ --entrypoint bash symbolicreasd05db995.azurecr.io/train:$VERSION-builder \
    -c '/workspace/strip_and_pack.sh'

for f in $(cat /workspace/filenames.txt); do
    if [[ $(file torch_shm_manager) == *"not strippeds"* ]]; then 
        echo hi;
    fi
done

echo -e "${GREEN}Build image...$NC"
docker build -f runner.Dockerfile -t symbolicreasd05db995.azurecr.io/train:$VERSION .

# Test it
echo -e "${GREEN}Test image...$NC"
docker run -w /workspace -v $PWD/..:/workspace --entrypoint /opt/conda/envs/pytorch-py37/bin/python symbolicreasd05db995.azurecr.io/train:$VERSION \
    ./ml/train.py -c real_world_problems/basics/dataset.yaml -v --data-size-limit 100 -n 1

rm -rf $tmp_dir