import argparse


class Parser(argparse.ArgumentParser):

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

        return args, model_parameters
