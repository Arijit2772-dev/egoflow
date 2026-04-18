from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, get_args, get_origin


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, obj: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(obj) if is_dataclass(obj) else obj
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_dataclass(path: str | Path, cls: type) -> Any:
    return from_dict(cls, read_json(path))


def from_dict(cls: type, data: Any) -> Any:
    if data is None:
        return None

    origin = get_origin(cls)
    args = get_args(cls)

    if origin in (list, tuple):
        inner = args[0] if args else Any
        converted = [from_dict(inner, item) for item in data]
        return tuple(converted) if origin is tuple else converted

    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {key: from_dict(value_type, value) for key, value in data.items()}

    if origin is not None and (origin is UnionType or str(origin) == "typing.Union"):
        non_none = [arg for arg in args if arg is not type(None)]
        if data is None or not non_none:
            return None
        return from_dict(non_none[0], data)

    if isinstance(cls, type) and issubclass(cls, Enum):
        return cls(data)

    if isinstance(cls, type) and is_dataclass(cls):
        hints = {field.name: field.type for field in fields(cls)}
        kwargs = {}
        for name, typ in hints.items():
            if name in data:
                kwargs[name] = from_dict(typ, data[name])
        return cls(**kwargs)

    return data
