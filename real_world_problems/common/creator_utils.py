from dataclasses import dataclass
from pathlib import Path
import yaml
import random
from tqdm import tqdm
import logging

from pycore import ScenarioProblems, Rule, Context, Symbol

logger = logging.getLogger(__name__)


@dataclass
class Replacer:
    pattern: str
    target: str
    padding: Symbol
    pad_size: int = 2


def collect(args, config_path: Path, factories, replacer: Replacer = None):
    loglevel = 'INFO' if args.verbose else args.log.upper()
    logging.basicConfig(
        level=logging.getLevelName(loglevel), format='%(message)s', datefmt='%I:%M:%S'
    )

    random.seed(0)

    me = Path(__file__).absolute()
    root = me.parents[2]

    with config_path.open() as f:
        config = yaml.safe_load(f)

        scenario_path = root / config['files']['scenario']
    with scenario_path.open() as f:
        scenario_data = yaml.safe_load(f)

        problems_path = Path(scenario_data['problems']['filename'])

    scenario_problems = ScenarioProblems()
    context = Context.standard()

    if not isinstance(factories, list):
        problems, used_idents = factories(**vars(args))
    else:
        problems = []
        used_idents = set()
        for factory in factories:
            new_problems, new_used_idents = factory(**vars(args))
            used_idents.update(new_used_idents)
            problems += new_problems
        used_idents = list(used_idents)
    problem_size = len(problems)

    # Take 10% as validation set
    validation_indices = set(random.sample(range(problem_size), k=problem_size // 10))

    for i, problem in tqdm(
        enumerate(problems),
        desc=f'Dumping to {problems_path}',
        smoothing=0.0,
        leave=False,
    ):
        rule = Rule.parse(context, problem, f'solving {i+1}')
        if replacer is not None:
            rule.replace_and_pad(pattern=replacer.pattern, target=replacer.pattern,
                                 pad_size=replacer.pad_size, pad_symbol=replacer.padding)
        if i in validation_indices:
            scenario_problems.add_to_validation(rule)
        else:
            scenario_problems.add_to_training(rule)

    logger.info(f'Used idents: {", ".join(used_idents)}')
    scenario_problems.add_additional_idents(list(set(used_idents)))
    logger.info(f'Dumping {problem_size} problems to {problems_path} ...')
    problems_path.parent.mkdir(exist_ok=True, parents=True)
    scenario_problems.dump(str(problems_path))


def create_term(operations, symbols):
    a = symbols[0]
    for b, operation in zip(symbols[1:], operations):
        if operation == '+':
            a = a + b
        elif operation == '-':
            a = a - b
        elif operation == '*':
            a = a * b
        elif operation == '/':
            if b == 0:
                raise ZeroDivisionError()
            a = a / b
        elif operation == '^':
            a = a ** b
        else:
            raise RuntimeError(f'Unsupported operator {operation}')

    return a
