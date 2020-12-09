#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging

from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator

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
        entry_script='./ml/t3_loop.py',
        script_params={
            '-c': args.config,
            '-v': '',
            '--tensorboard-dir': 'outputs/tensorboard',
            '--policy-last': '',
            '--log': 'info',
            '--fresh-model': '',
            '--training-num-epochs': '100',
        },
        source_directory=Path(__file__).absolute().parents[1],
        user_managed=True,
        process_count_per_node=1,
        custom_docker_image=docker_image,
        use_gpu=False
    )

    run = experiment.submit(estimator, tags={k: v for k, v in [i.split(':') for i in args.tags]})
    print(run.get_portal_url())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Submit an Azure ML t3 job')

    add_default_parsers(parser)

    parser.add_argument('--config', default='real_world_problems/number_crunching/config.yaml')

    parser.add_argument('--tags', nargs='*', default=[])
    parser.add_argument('--name', default='t3-loop')

    main(parser.parse_args())
