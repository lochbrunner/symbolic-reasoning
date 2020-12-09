import argparse
import logging


def setup_logging(args: argparse.Namespace):
    loglevel = 'INFO' if args.verbose else args.log.upper()
    log_format = '%(message)s'
    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format=log_format,
        datefmt='%I:%M:%S'
    )


def add_default_parsers(parser: argparse.ArgumentParser):
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    parser.add_argument('--docker-registry', default='symbolicreasd05db995.azurecr.io')
    parser.add_argument('--docker-repository', default='train')
    parser.add_argument('--docker-image', default='13-builder')
    parser.add_argument('--compute-target', default='cpucore8')

    parser.add_argument('--workspace-name', default='symbolic-reasoning-aml')
    parser.add_argument('--subscription-id', default='4c2ff317-b3e5-4302-b705-688087514d74')
    parser.add_argument('--resource-group', default='symbolic-reasoning')

    return parser
