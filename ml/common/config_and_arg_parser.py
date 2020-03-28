import argparse


class Parser(argparse.ArgumentParser):

    def __init__(self, *config_option_strings, config_name='config', loader, **kwargs):
        '''
        '''
        super(Parser, self).__init__(**kwargs)
        self.config_option_strings = config_option_strings
        self.loader = loader
        self.config_name = config_name

    def parse_args(self, **kwargs):
        config_file_parser = argparse.ArgumentParser(add_help=False)
        config_file_parser.add_argument(*self.config_option_strings,
                                        dest='config_file')
        args, remaining_argv = config_file_parser.parse_known_args(**kwargs)
        config_file = args.config_file
        if config_file is None:
            return super(Parser, self).parse_args(**kwargs)

        defaults = self.loader(args.config_file)
        normalized_defaults = {k.replace('-', '_'): defaults[k] for k in defaults}

        self.set_defaults(**normalized_defaults)
        kwargs['args'] = remaining_argv
        args = super(Parser, self).parse_args(**kwargs)
        setattr(args, self.config_name, config_file)
        return args
