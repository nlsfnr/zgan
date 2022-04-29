from __future__ import annotations
import yaml
from typing import Dict, Any
from pathlib import Path


class AttrDict(Dict[str, Any]):

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            if key not in self:
                raise e
            return self[key]

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AttrDict:
        return AttrDict(**{k: cls.from_dict(v) if isinstance(v, dict) else v
                           for k, v in d.items()})

    @classmethod
    def from_yaml(cls, path: Path) -> AttrDict:
        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d)
