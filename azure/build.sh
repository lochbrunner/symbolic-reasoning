#!/usr/bin/env bash

set -ex

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

BUILDER_VERSION=8
VERSION=${VERSION:-"10"}
REGISTRY=symbolicreasd05db995.azurecr.io
IMAGE=train

# Creating temporary folder
tmp_dir=/tmp/$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 10 ; echo '')
trace_file=$tmp_dir/strace.txt
docker_context=$tmp_dir/docker
mkdir -p $docker_context


mkdir -p workspace
echo -e "${GREEN}Tracing...$NC"
docker run -w /workspace -v $PWD/..:/workspace -v $tmp_dir:/tmp/ --entrypoint strace $REGISTRY/$IMAGE:$BUILDER_VERSION-builder \
 -e trace=open,openat -f -o /tmp/$(basename $trace_file) \
 ./ml/train.py -c real_world_problems/basics/dataset.yaml -v --data-size-limit 100 -n 1 > /dev/null

# echo -e "${GREEN}Resolving globs of additional files...$NC"
# docker run -w /workspace -v $PWD/..:/workspace --entrypoint /workspace/azure/expand_additional_files.py \ 
#     $REGISTRY/$IMAGE:$BUILDER_VERSION-builder \
#     -i /workspace/azure/additional_files.txt -O /workspace/workspace/resolved_addional_files.txt

echo -e "${GREEN}Creating complete file list...$NC"
./extract_used_files.py $trace_file -O $tmp_dir/filenames.txt -a additional_files.txt

echo -e "${GREEN}Packing files...$NC"
docker run -w /workspace -v $PWD/:/workspace/ -v $tmp_dir:/tmp/ --entrypoint bash $REGISTRY/$IMAGE:$BUILDER_VERSION-builder \
    -c '/workspace/strip_and_pack.sh /tmp'

# for f in $(cat /workspace/filenames.txt); do
#     if [[ $(file torch_shm_manager) == *"not strippeds"* ]]; then 
#         echo hi;
#     fi
# done

echo -e "${GREEN}Build image...$NC"
cp runner.Dockerfile $docker_context/Dockerfile
docker build -t $REGISTRY/$IMAGE:$VERSION $docker_context

# Test it
echo -e "${GREEN}Test image...$NC"
docker run -w /workspace -v $PWD/..:/workspace --entrypoint ./ml/train.py $REGISTRY/$IMAGE:$VERSION \
    -c real_world_problems/basics/dataset.yaml -v --data-size-limit 100 -n 1

echo -e ${RED}Please remove $tmp_dir$NC
# rm -rf $tmp_dir
