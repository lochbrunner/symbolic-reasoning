#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging

from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import BayesianParameterSampling, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice

from lib.common import add_default_parsers, setup_logging

logger = logging.getLogger(__name__)


def main(args):
    setup_logging(args)

    ws = Workspace.get(name=args.workspace_name,
                       subscription_id=args.subscription_id,
                       resource_group=args.resource_group)

    docker_image = f'{args.docker_registry}/{args.docker_repository}:{args.docker_image}'

    experiment = Experiment(workspace=ws, name=args.name)

    estimator = Estimator(
        compute_target=args.compute_target,
        entry_script='./ml/solve.py',
        script_params={
            '-c': args.config,
            '-v': '',
            '--policy-last': '',
            '--log': 'info',
            '--fresh-model': '',
        },
        source_directory=Path(__file__).absolute().parents[1],
        user_managed=True,
        process_count_per_node=1,
        custom_docker_image=docker_image,
        use_gpu=False
    )

    ps = BayesianParameterSampling(
        {
            '--problems-max_fit_results': choice(list(range(1000, 6000, 1000))),
            '--max-size': choice(list(range(10, 15))),
        }
    )

    hdc = HyperDriveConfig(estimator=estimator,
                           hyperparameter_sampling=ps,
                           primary_metric_name='rate/success',
                           primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                           max_total_runs=20,
                           max_concurrent_runs=2)

    experiment = Experiment(workspace=ws, name=args.name)

    run = experiment.submit(hdc, tags={k: v for k, v in [i.split(':') for i in args.tags]})
    print(run.get_portal_url())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Submit a solve hyperdrive job to Azure')
    parser.add_argument('--config', default='real_world_problems/number_crunching/config.yaml')

    add_default_parsers(parser)

    parser.add_argument('--tags', nargs='*', default=[])
    parser.add_argument('--name', default='solve')

    main(parser.parse_args())
