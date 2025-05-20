import argparse 
from mode import Mode
from typing import Type
from console import Console

class App:
    modes: dict[str, Mode] = {}

    def __init__(self, console: Console):
        self.parser = argparse.ArgumentParser()
        self.subparser = self.parser.add_subparsers(dest="mode", required=True)
        self.console = console

    def use(self, name: str, mode: Type[Mode]):
        self.modes[name] = mode
        mode.add_subparser(name, self.subparser)

    def run(self):
        args = self.parser.parse_args()
        dargs = {k: v for k, v in args.__dict__.items() if k != "mode"}

        mode = self.modes[args.mode](self.console, **dargs)
        mode.run()