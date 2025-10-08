import sys
import types
import pytest
from recollex.utils import load_callable

# Build a dummy module in sys.modules
mod = types.ModuleType("dummy_mod_errs")
class NoCall:
    def __init__(self): pass
mod.NoCall = NoCall
sys.modules["dummy_mod_errs"] = mod

def test_load_callable_bad_spec_raises():
    with pytest.raises(ValueError):
        load_callable("not_a_module_or_spec")

def test_load_callable_missing_attr_raises():
    with pytest.raises(AttributeError):
        load_callable("dummy_mod_errs:Missing")

def test_load_callable_class_not_callable_instance_raises():
    with pytest.raises(TypeError):
        load_callable("dummy_mod_errs:NoCall")

def test_load_callable_non_callable_object_raises():
    with pytest.raises(TypeError):
        load_callable(object())
