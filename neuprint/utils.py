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


def where_expr(field, values, regex=False, name='n'):
    """
    Return an expression to match a particular
    field against a list of values, to be used
    within the WHERE clause.
    """
    assert not regex or len(values) <= 1, \
        f"Can't use regex mode with more than one value: {values}"

    if len(values) == 0:
        return ""

    if len(values) > 1:
        return f"{name}.{field} in {[*values]}"

    if regex:
        return f"{name}.{field} =~ '{values[0]}'"

    if isinstance(values[0], str):
        return f"{name}.{field} = '{values[0]}'"

    return f"{name}.{field} = {values[0]}"


