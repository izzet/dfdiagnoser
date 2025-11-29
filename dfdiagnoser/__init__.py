from typing import Union
from .input import CheckpointInput
from .output import ConsoleOutput, FileOutput

InputType = Union[CheckpointInput]
OutputType = Union[ConsoleOutput, FileOutput]

__all__ = ["InputType", "OutputType",
           "CheckpointInput", "ConsoleOutput", "FileOutput"]
