import inspect
import functools
from collections.abc import Iterable


def make_iterable(x):
    """
    If ``x`` is already a list or array, return it unchanged.
    If ``x`` is ``None``, return an empty list ``[]``.
    Otherwise, wrap it in a list.
    """
    if x is None:
        return []
    if isinstance(x, Iterable) and not isinstance(x, str):
        return x
    else:
        return [x]


def make_args_iterable(argnames):
    """
    Returns a decorator.
    For the given argument names, the decorator converts the
    arguments into iterables via ``make_iterable()``.
    """
    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(f, *args, **kwargs)
            for name in argnames:
                callargs[name] = make_iterable(callargs[name])
            return f(**callargs)

        wrapper.__signature__ = inspect.signature(f)
        return wrapper

    return decorator


