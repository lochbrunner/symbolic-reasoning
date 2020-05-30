#!/usr/bin/env python3

import argparse

from azureml.core import Workspace, Experiment, RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.datastore import Datastore
from azureml.core.environment import Environment, PythonSection
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline


import os


def main(args):
    ws = Workspace.get(name=args.workspace_name,
                       subscription_id=args.subscription_id,
                       resource_group=args.resource_group)

    runconfig = RunConfiguration()

    runconfig.environment.python.user_managed_dependencies = True
    runconfig.environment.python.interpreter_path = '/opt/conda/envs/pytorch-py37/bin/python'
    runconfig.environment.docker.enabled = True
    runconfig.environment.docker.base_image = f'{args.docker_registry}/{args.docker_repository}:{args.docker_image}'
    runconfig.environment.docker.base_image_registry.address = args.docker_registry
    runconfig.target = ws.compute_targets[args.compute_target]

    generated_datastore = Datastore.get(ws, 'generated')
    generated_datastore = DataReference(
        datastore=generated_datastore,
        data_reference_name='generated',
        path_on_datastore=args.trainings_data)

    train_step = PythonScriptStep(
        script_name='ml/train.py',
        runconfig=runconfig,
        arguments=['-c', args.dataset, '-v',
                   '--bag-filename', generated_datastore,
                   '--save-model', 'outputs/bag-basic_parameter_search.sp',
                   '--statistics', 'outputs/training-statistics.yaml'],
        inputs=[generated_datastore],
        outputs=[],
        source_directory=os.path.realpath(os.path.dirname(__file__)+'/..'),
        allow_reuse=False
    )
    pipeline = Pipeline(workspace=ws, steps=[train_step])
    experiment = Experiment(workspace=ws, name='training')

    experiment.submit(pipeline, tags={k: v for k, v in [i.split(':') for i in args.tags]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Submit Azure ML trainings job')
    parser.add_argument('--docker-registry', default='symbolicreasd05db995.azurecr.io')
    parser.add_argument('--docker-repository', default='train')
    parser.add_argument('--docker-image', default='6')
    parser.add_argument('--compute-target', default='cpucore8')

    parser.add_argument('--trainings-data', default='experiments/bag-basic.bin')
    parser.add_argument('--dataset', default='real_world_problems/basics/dataset.yaml')

    parser.add_argument('--tags', nargs='*', default=[])
    parser.add_argument('--name', default='training')

    parser.add_argument('--workspace-name', default='symbolic-reasoning-aml')
    parser.add_argument('--subscription-id', default='4c2ff317-b3e5-4302-b705-688087514d74')
    parser.add_argument('--resource-group', default='symbolic-reasoning')

    main(parser.parse_args())
