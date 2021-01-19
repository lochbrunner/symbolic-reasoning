import argparse
import yaml
from pathlib import Path
from typing import List, Dict
import unittest
from fnmatch import fnmatch


def unroll_nested_dict(dictionary, prefix='--', parents=None, seperator='-'):
    for key, value in dictionary.items():
        if parents is None:
            n_parents = (key,)
        else:
            n_parents = (*parents, key)
        if type(value) in (str, int, float, list, bool):
            yield (f'{prefix}{key}', value, n_parents)
        elif type(value) is dict:
            for pair in unroll_nested_dict(value, f'{prefix}{key}{seperator}', n_parents):
                yield pair
        else:
            raise NotImplementedError(f'Loading config does not support type {type(value)}')


class TestUnrollNestedDict(unittest.TestCase):
    def test_unroll(self):
        A = {
            "a": 12,
            "b": [2, 4],
            "c": {
                "d": 1,
                "e": {
                    "f": "4",
                    "g": 2
                }
            }
        }

        unrolled = list(unroll_nested_dict(A))
        expected = [
            ('--a', 12, ('a',)),
            ('--b', [2, 4], ('b',)),
            ('--c-d', 1, ('c', 'd')),
            ('--c-e-f', '4', ('c', 'e', 'f')),
            ('--c-e-g', 2, ('c', 'e', 'g'))
        ]
        self.assertCountEqual(unrolled, expected)


def create_nested_namespace(orig: Dict, flat_args: argparse.Namespace, parent='', seperator='-'):
    namespace = argparse.Namespace()

    def set_field(fields, value):
        current_ns = namespace
        fields = [field.replace('-', '_') for field in fields]
        for field in fields[:-1]:
            if not hasattr(current_ns, field):
                setattr(current_ns, field, argparse.Namespace())
            current_ns = getattr(current_ns, field)

        if hasattr(current_ns, fields[-1]):
            raise RuntimeError(f'Field {fields} is already set, with {getattr(current_ns, fields[-1])}')
        setattr(current_ns, fields[-1], value)

    for key, default_value, fields in unroll_nested_dict(orig, parents=parent, seperator=seperator):
        key = key[2:].replace('-', '_')
        if hasattr(flat_args, key):
            value = getattr(flat_args, key)
        else:
            value = default_value
        set_field(fields, value)

    return namespace


class TestCreateNestedDict(unittest.TestCase):
    def test_create_namespace(self):
        orig_dict = {
            "a": 12,
            "b": [2, 4],
            "c": {
                "d": 1,
                "e": {
                    "f": "4",
                    "g": 2
                }
            },
            "h-i": 24,
        }

        parser = argparse.ArgumentParser()
        for name, value, _ in unroll_nested_dict(orig_dict):
            parser.add_argument(name, default=value, type=type(value))

        flat_args = parser.parse_args(['--a', '3'])

        nested_args = create_nested_namespace(orig_dict, flat_args)

        self.assertEqual(nested_args.a, 3)
        self.assertEqual(nested_args.b, [2, 4])
        self.assertEqual(nested_args.c.d, 1)
        self.assertEqual(nested_args.c.e.f, "4")
        self.assertEqual(nested_args.c.e.g, 2)
        self.assertEqual(nested_args.h_i, 24)


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, short_config_name='-c', long_config_name='--config-file',
                 domain=None, seperator='-', exclude=None, **kwargs):
        '''
        Creates a wrapper around the original argparse.ArgumentParse

        additional params:
            * domain: the items of the top level node with this name 
                        in the config get moved at top level
            * exclude: globs of the path which should be excluded
        '''

        super(ArgumentParser, self).__init__(**kwargs)
        self.domain = domain
        self.seperator = seperator
        self.exclude = exclude
        self.config_name = (short_config_name, long_config_name)

    def parse_args(self, args=None, namespace=None):
        config_file_parser = argparse.ArgumentParser()
        config_file_parser.add_argument(*self.config_name, type=Path, required=True)
        args, remaining_argv = config_file_parser.parse_known_args(args)
        config_file = getattr(args, self.config_name[1][2:].replace('-', '_'))
        with config_file.open() as f:
            config = yaml.full_load(f)

        # Move the domain node to to top
        if self.domain is not None:
            domain_node = config[self.domain]
            domain_node_keys = domain_node.keys()
            del config[self.domain]
            config = {**config, **domain_node}

        parser = argparse.ArgumentParser()
        for name, value, _ in unroll_nested_dict(config, seperator=self.seperator):
            if fnmatch(name, self.exclude):
                continue
            dtype = type(value)
            if dtype is bool:
                if value:
                    name = '--no' + name[1:]
                    parser.add_argument(name, action='store_false', default=True)
                else:
                    parser.add_argument(name, action='store_true')
            elif dtype is list:
                if len(value) > 0:
                    parser.add_argument(name, default=value, nargs='+', type=type(value[0]))
                else:
                    parser.add_argument(name, default=value, nargs='+')
            else:
                if dtype is str and (name.endswith("-filename") or name.endswith("-data")):
                    parser.add_argument(name, default=value, type=Path)
                else:
                    parser.add_argument(name, default=value, type=type(value))

        flat_args, remaining_argv = parser.parse_known_args(remaining_argv)

        config_args = create_nested_namespace(config, flat_args, seperator=self.seperator)

        # Move the domain node back
        if self.domain is not None:
            domain_node = argparse.Namespace()
            for key in domain_node_keys:
                key = key.replace('-', '_')
                node = getattr(config_args, key)
                setattr(domain_node, key, node)
                delattr(config_args, key)
            setattr(config_args, self.domain, domain_node)

        self_args = super(ArgumentParser, self).parse_args(remaining_argv)
        self_args.config_file = config_file

        return config_args, self_args


class Parser(argparse.ArgumentParser):
    '''Deprecated: Use `ArgumentParser` instead!'''

    def __init__(self, *config_option_strings, config_name='config', loader, **kwargs):
        '''
        '''
        super(Parser, self).__init__(**kwargs)
        self.config_option_strings = config_option_strings
        self.loader = loader
        self.config_name = config_name

    def parse_args(self, args=None, namespace=None):
        config_file_parser = argparse.ArgumentParser(add_help=False)
        config_file_parser.add_argument(*self.config_option_strings,
                                        dest='config_file')
        args, remaining_argv = config_file_parser.parse_known_args(args, namespace)
        config_file = args.config_file
        if config_file is None:
            return super(Parser, self).parse_args(args, namespace)

        defaults = self.loader(args.config_file)
        normalized_defaults = {k.replace('-', '_'): defaults[k] for k in defaults}
        self.set_defaults(**normalized_defaults)
        args, remaining_argv = super(Parser, self).parse_known_args(args=remaining_argv, namespace=namespace)
        setattr(args, self.config_name, config_file)

        # model parameter (hyper parameters)
        if 'model-parameter' in defaults:
            default_model_parameters = defaults['model-parameter']
            model_parameter_parser = argparse.ArgumentParser()
            for mp_name in default_model_parameters:
                default = default_model_parameters[mp_name]
                if isinstance(default, list):
                    dtype = type(default[0])
                else:
                    dtype = type(default)
                model_parameter_parser.add_argument('--' + mp_name.replace('_', '-'), dest=mp_name,
                                                    default=default, type=dtype)
            mp_args = model_parameter_parser.parse_args(remaining_argv)
            model_parameters = {}
            for mp_name in default_model_parameters:
                model_parameters[mp_name] = getattr(mp_args, mp_name)
        else:
            model_parameters = None

        return args, model_parameters, defaults


if __name__ == '__main__':
    unittest.main()
