import inspect
import numpy as np
from neuprint.utils import make_iterable, make_args_iterable


def test_make_iterable():
    assert make_iterable(None) == []
    assert make_iterable([None]) == [None]

    assert make_iterable(1) == [1]
    assert make_iterable([1]) == [1]

    assert isinstance(make_iterable(np.array([1,2,3])), np.ndarray)


def test_make_args_iterable():
    
    @make_args_iterable(['a', 'c', 'd'])
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
