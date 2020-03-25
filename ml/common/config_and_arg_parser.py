import argparse


class Parser(argparse.ArgumentParser):

    def __init__(self, *config_option_strings, loader, **kwargs):
        '''
        '''
        super(Parser, self).__init__(**kwargs)
        self.config_option_strings = config_option_strings
        self.loader = loader

    def parse_args(self, **kwargs):
        config_file_parser = argparse.ArgumentParser(add_help=False, parents=[self])
        config_file_parser.add_argument(*self.config_option_strings,
                                        dest='config_file')
        args, remaining_argv = config_file_parser.parse_known_args(**kwargs)

        if args.config_file is None:
            return super(Parser, self).parse_args(**kwargs)

        defaults = self.loader(args.config_file)
        normalized_defaults = {k.replace('-', '_'): defaults[k] for k in defaults}

        self.set_defaults(**normalized_defaults)
        kwargs['args'] = remaining_argv
        return super(Parser, self).parse_args(**kwargs)
