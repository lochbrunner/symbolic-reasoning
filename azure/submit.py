#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging

from azureml.core import Workspace, Experiment
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal
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
        script_params={'-c': args.dataset,
                       '-v': '',
                       '--bag-filename': str(path_on_compute),
                       '--save-model': 'outputs/bag-basic_parameter_search.sp',
                       '--report-rate': '10',
                       '--statistics': 'outputs/training-statistics.yaml'},
        source_directory=Path(__file__).parents[1],
        user_managed=True,
        process_count_per_node=1,
        custom_docker_image=docker_image,
        use_gpu=False
    )

    ps = GridParameterSampling(
        {
            '--embedding-size': choice([16, 24, 32, 48]),
            '--use-props': choice([1, 0]),
        }
    )

    hdc = HyperDriveConfig(estimator=estimator,
                           hyperparameter_sampling=ps,
                           primary_metric_name='final_error',
                           primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                           max_total_runs=20,
                           max_concurrent_runs=2)

    experiment = Experiment(workspace=ws, name='training')

    run = experiment.submit(hdc, tags={k: v for k, v in [i.split(':') for i in args.tags]})
    print(run.get_portal_url())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Submit Azure ML trainings job')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    parser.add_argument('--docker-registry', default='symbolicreasd05db995.azurecr.io')
    parser.add_argument('--docker-repository', default='train')
    parser.add_argument('--docker-image', default='7-builder')
    parser.add_argument('--compute-target', default='cpucore8')

    parser.add_argument('--trainings-data', default='experiments/bag-basic.bin', type=Path)
    parser.add_argument('--dataset', default='real_world_problems/basics/dataset.yaml', type=Path)

    parser.add_argument('--tags', nargs='*', default=[])
    parser.add_argument('--name', default='training')

    parser.add_argument('--workspace-name', default='symbolic-reasoning-aml')
    parser.add_argument('--subscription-id', default='4c2ff317-b3e5-4302-b705-688087514d74')
    parser.add_argument('--resource-group', default='symbolic-reasoning')

    main(parser.parse_args())
