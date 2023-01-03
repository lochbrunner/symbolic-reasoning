#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging

from azureml.core import Workspace, Experiment, Run

logger = logging.getLogger(__name__)


def main(args):
    ws = Workspace.get(name=args.workspace_name,
                       subscription_id=args.subscription_id,
                       resource_group=args.resource_group)

    experiment = Experiment(workspace=ws, name=args.name)

    run = Run(experiment, args.run_id)
    for c in run.get_children():
        path = args.output_directory / c.id
        c.download_files(output_directory=path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Download experiment run results')
    parser.add_argument('--workspace-name', default='symbolic-reasoning-aml')
    parser.add_argument('--subscription-id', default='4c2ff317-b3e5-4302-b705-688087514d74')
    parser.add_argument('--resource-group', default='symbolic-reasoning')

    parser.add_argument('--name', default='training')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-directory', required=True, type=Path)

    main(parser.parse_args())
