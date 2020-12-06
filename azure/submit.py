#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging

from azureml.core import Workspace, Experiment
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import BayesianParameterSampling, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice

logger = logging.getLogger(__name__)


def main(args):
    loglevel = 'INFO' if args.verbose else args.log.upper()
    log_format = '%(message)s'
    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format=log_format,
        datefmt='%I:%M:%S'
    )

    ws = Workspace.get(name=args.workspace_name,
                       subscription_id=args.subscription_id,
                       resource_group=args.resource_group)

    generated_datastore = Datastore.get(ws, 'generated')
    generated_dr = DataReference(
        datastore=generated_datastore,
        data_reference_name='generated',
        path_on_datastore=args.trainings_data)

    docker_image = f'{args.docker_registry}/{args.docker_repository}:{args.docker_image}'

    logger.info(f'Using Docker image: {docker_image}')
    dir_on_compute = Path('/mnt/data')
    path_on_compute = dir_on_compute / args.trainings_data
    estimator = Estimator(
        compute_target=args.compute_target,
        entry_script='./ml/train.py',
        inputs=[generated_dr.as_download(path_on_compute=dir_on_compute)],
        script_params={'-c': args.config,
                       '-v': '',
                       '--training-solver-filename': str(path_on_compute),
                       '--files-solver-trainings-data': str(path_on_compute),
                       '--files-model': 'out/number-crunching.sp',
                       '--files-training-statistics': 'out/training-statistics.yaml',
                       '--use-solved-problems': '',
                       '--tensorboard': '',
                       '--training-num-epochs': '500',
                       '-r': '10',
                       '--create-fresh-model': ''},
        source_directory=Path(__file__).absolute().parents[1],
        user_managed=True,
        process_count_per_node=1,
        custom_docker_image=docker_image,
        use_gpu=False
    )

    ps = BayesianParameterSampling(
        {
            '--training-model-parameter-embedding_size': choice([32, 48]),
            '--training-model-parameter-hidden_layers': choice([1, 2, 3]),
            '--training-batch-size': choice([16, 32, 48]),
            '--training-value-loss-weight': choice([0.5, 0.25]),
        }
    )

    hdc = HyperDriveConfig(estimator=estimator,
                           hyperparameter_sampling=ps,
                           primary_metric_name='exact (np) [5]',
                           primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                           max_total_runs=50,
                           max_concurrent_runs=4)

    experiment = Experiment(workspace=ws, name=args.name)

    run = experiment.submit(hdc, tags={k: v for k, v in [i.split(':') for i in args.tags]})
    print(run.get_portal_url())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Submit Azure ML trainings job')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    parser.add_argument('--docker-registry', default='symbolicreasd05db995.azurecr.io')
    parser.add_argument('--docker-repository', default='train')
    parser.add_argument('--docker-image', default='12-builder')
    parser.add_argument('--compute-target', default='cpucore8   ')

    parser.add_argument('--trainings-data', default='experiments/number-crunching/solver-trainings-data.bin', type=Path)
    parser.add_argument('--config', default='real_world_problems/number_crunching/config.yaml', type=Path)

    parser.add_argument('--tags', nargs='*', default=[])
    parser.add_argument('--name', default='training')

    parser.add_argument('--workspace-name', default='symbolic-reasoning-aml')
    parser.add_argument('--subscription-id', default='4c2ff317-b3e5-4302-b705-688087514d74')
    parser.add_argument('--resource-group', default='symbolic-reasoning')

    main(parser.parse_args())
