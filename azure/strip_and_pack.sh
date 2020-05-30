#!/bin/bash

for f in $(cat /workspace/workspace/filenames.txt); do
    if [[ $(file torch_shm_manager) == *"not stripped"* ]]; then 
        strip -s $i 
    fi
done

tar -C / -Pcvhf /workspace/files.tar $(cat /workspace/workspace/filenames.txt)