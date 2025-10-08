from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional, Mapping


def load_callable(spec_or_obj: Any, ctor_kwargs: Optional[Dict[str, Any]] = None) -> Callable[..., Any]:
    """
    Accept a function, an instance, a class with __call__, or an import spec "pkg.mod:attr" (preferred) or "pkg.mod.attr".
    If a class is provided, it will be instantiated with ctor_kwargs (defaults to {}).
    """
    if callable(spec_or_obj) and not inspect.isclass(spec_or_obj):
        # function or instance with __call__
        return spec_or_obj

    if isinstance(spec_or_obj, str):
        if ":" in spec_or_obj:
            mod, name = spec_or_obj.split(":", 1)
        else:
            mod, _, name = spec_or_obj.rpartition(".")
        if not mod or not name:
            raise ValueError(f"Expected import spec 'package.module:attr' or 'package.module.attr'; got '{spec_or_obj}'")
        obj = importlib.import_module(mod)
        # Support nested attributes after import (e.g., "pkg.mod:Class.method")
        for part in name.split("."):
            obj = getattr(obj, part)
    else:
        obj = spec_or_obj

    if inspect.isclass(obj):
        inst = obj(**(ctor_kwargs or {}))
        if not callable(inst):
            raise TypeError(f"Instantiated '{obj.__name__}' but the instance is not callable")
        return inst

    if callable(obj):
        return obj

    raise TypeError(f"Object '{obj}' of type {type(obj).__name__} is not callable")


def resolve_hooks(
    hooks: Mapping[str, Any],
    ctor_kwargs: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, Callable[..., Any]]:
    """
    Resolve a mapping of hook names -> spec (function/instance/class/dotted path).
    Optional per-hook ctor kwargs: {name: {...}} for classes.
    """
    resolved: Dict[str, Callable[..., Any]] = {}
    per_hook_kwargs = ctor_kwargs or {}
    for name, spec in hooks.items():
        resolved[name] = load_callable(spec, ctor_kwargs=per_hook_kwargs.get(name))
    return resolved
