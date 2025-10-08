import sys
import types

def test_load_callable_and_resolve_hooks(monkeypatch):
    # Build a dummy module in sys.modules
    mod = types.ModuleType("dummy_mod")
    def foo(x): return x + 1
    class Adder:
        def __init__(self, inc): self.inc = inc
        def __call__(self, x): return x + self.inc
    mod.foo = foo
    mod.Adder = Adder
    sys.modules["dummy_mod"] = mod

    from recollex.utils import load_callable, resolve_hooks

    # Load function by spec
    fn = load_callable("dummy_mod:foo")
    assert callable(fn) and fn(2) == 3

    # Load class by spec with ctor kwargs; returns callable instance
    adder = load_callable("dummy_mod:Adder", ctor_kwargs={"inc": 5})
    assert callable(adder) and adder(7) == 12

    # Passing a callable object returns it
    fn2 = load_callable(foo)
    assert fn2 is foo and fn2(1) == 2

    # resolve_hooks mixes function and class specs, with per-hook kwargs
    hooks = resolve_hooks(
        {"score": "dummy_mod:foo", "cand": "dummy_mod:Adder"},
        ctor_kwargs={"cand": {"inc": 2}},
    )
    assert callable(hooks["score"]) and hooks["score"](3) == 4
    assert callable(hooks["cand"]) and hooks["cand"](3) == 5
