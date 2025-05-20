from abc import ABC
from abc import abstractmethod
from console import Console
from argparse import _SubParsersAction

class Mode(ABC):
    def __init__(
        self, 
        console: Console):
        self.console = console

    @staticmethod
    @abstractmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        pass

    @abstractmethod
    def run(self):
        pass