#!/bin/bash

context=$1
filnames_file=$context/filenames.txt
archive_path=$context/docker/files.tar

echo "using file $filnames_file" 

for f in $(cat $filnames_file); do
    if [[ $(file torch_shm_manager) == *"not stripped"* ]]; then 
        strip -s $f 
    fi
done

tar -C / -Pcvhf $archive_path $(cat $filnames_file)