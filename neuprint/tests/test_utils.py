import inspect
import numpy as np
from neuprint.utils import ensure_list, ensure_list_args


def test_ensure_list():
    assert ensure_list(None) == []
    assert ensure_list([None]) == [None]

    assert ensure_list(1) == [1]
    assert ensure_list([1]) == [1]

    assert isinstance(ensure_list(np.array([1,2,3])), list)


def test_ensure_list_args():

    @ensure_list_args(['a', 'c', 'd'])
    def f(a, b, c, d='d', *, e=None):
        return (a,b,c,d,e)

    # Must preserve function signature
    spec = inspect.getfullargspec(f)
    assert spec.args == ['a', 'b', 'c', 'd']
    assert spec.defaults == ('d',)
    assert spec.kwonlyargs == ['e']
    assert spec.kwonlydefaults == {'e': None}

    # Check results
    assert f('a', 'b', 'c', 'd') == (['a'], 'b', ['c'], ['d'], None)
