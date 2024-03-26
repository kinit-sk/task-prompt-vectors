from transformers import HfArgumentParser
from typing import Tuple, NewType, Any
from pathlib import Path
import tomllib

DataClass = NewType("DataClass", Any)


class ArgumentParser(HfArgumentParser):
    def parse_toml_file(
        self, toml_file: str, allow_extra_keys: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a toml file and populating the
        dataclass types.

        Args:
            toml_file (`str` or `os.PathLike`):
                File name of the toml file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """

        with open(Path(toml_file), "rb") as f:
            data = tomllib.load(f)

        outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
        return tuple(outputs)
