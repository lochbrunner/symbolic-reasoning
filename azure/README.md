# Azure

## Building Docker Image

When stripping everything out but the traced files, there is a problem with the directories:

* You can not pack a directory and its child directories in a tar file. Extracting it in a docker container results in corrupted files.
* When the python interpreter launches it lists the content of a directory without opening all the files.

Stripping binaries

## Submitting

```zsh
# From repository root
azure/submit.py --tags reason:test-slim
```

## Open points

* How the get the filenames used by Azureml?