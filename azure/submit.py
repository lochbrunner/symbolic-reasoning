#!/usr/bin/env python3

from azureml.core import Workspace, Experiment, RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.datastore import Datastore
from azureml.core.environment import Environment, PythonSection
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline


import os

# ws = Workspace.from_config()
ws = Workspace.get(name="symbolic-reasoning-aml",
                   subscription_id='4c2ff317-b3e5-4302-b705-688087514d74',
                   resource_group='symbolic-reasoning')

runconfig = RunConfiguration()


runconfig.environment.python.user_managed_dependencies = True
runconfig.environment.python.interpreter_path = '/opt/conda/envs/pytorch-py37/bin/python'
runconfig.environment.docker.enabled = True
runconfig.environment.docker.base_image = 'symbolicreasd05db995.azurecr.io/train:4'
runconfig.environment.docker.base_image_registry.address = 'symbolicreasd05db995.azurecr.io'
runconfig.target = ws.compute_targets['cpucore2']

generated_datastore = Datastore.get(ws, 'generated')
generated_datastore = DataReference(
    datastore=generated_datastore,
    data_reference_name='generated',
    path_on_datastore='experiments/bag-basic.bin')

train_step = PythonScriptStep(
    script_name="ml/train.py",
    runconfig=runconfig,
    arguments=['-c', 'real_world_problems/basics/dataset.yaml', '-v',
               '--bag-filename', generated_datastore,
               '--save-model', 'outputs/bag-basic_parameter_search.sp',
               '--statistics', 'outputs/training-statistics.yaml'],
    inputs=[generated_datastore],
    outputs=[],
    source_directory=os.path.realpath(os.path.dirname(__file__)+'/..'),
    allow_reuse=False
)
pipeline = Pipeline(workspace=ws, steps=[train_step])
experiment = Experiment(workspace=ws, name="training")

run = experiment.submit(pipeline, tags={'reason': 'first-try'})
