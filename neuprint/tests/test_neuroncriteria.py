from textwrap import dedent

import pytest
import numpy as np
import pandas as pd

from neuprint import Client, default_client, set_default_client, NeuronCriteria as NC, NotNull, IsNull
from neuprint.queries.neuroncriteria import where_expr
from neuprint.tests import NEUPRINT_SERVER, DATASET


@pytest.fixture(scope='module')
def client():
    c = Client(NEUPRINT_SERVER, DATASET)
    set_default_client(c)
    assert default_client() == c
    return c


def test_NeuronCriteria(client):
    assert NC(bodyId=1).bodyId == [1]
    assert NC(bodyId=[1,2,3]).bodyId == [1,2,3]

    # It's important that bodyIds and ROIs are stored as plain lists,
    # since we naively serialize them into Cypher queries with that assumption.
    assert NC(bodyId=np.array([1,2,3])).bodyId == [1,2,3]
    assert NC(bodyId=pd.Series([1,2,3])).bodyId == [1,2,3]

    ##
    ## basic_exprs()
    ##
    assert NC(bodyId=123).basic_exprs() == ["n.bodyId = 123"]
    assert NC('m', bodyId=123).basic_exprs() == ["m.bodyId = 123"]
    assert NC(bodyId=[123, 456]).basic_exprs() == ["n.bodyId in [123, 456]"]

    assert NC(type='foo.*').regex
    assert not NC(type='foo').regex
    assert NC(instance='foo.*').regex
    assert not NC(instance='foo').regex

    # Cell types really contain parentheses sometimes,
    # so we don't want to automatically upgrade to regex mode for parentheses.
    assert not NC(type='foo(bar)').regex
    assert not NC(instance='foo(bar)').regex

    assert NC(instance="foo").basic_exprs() == ["n.instance = 'foo'"]
    assert NC(instance="foo", regex=True).basic_exprs() == ["n.instance =~ 'foo'"]
    assert NC(instance=["foo", "bar"]).basic_exprs() == ["n.instance in ['foo', 'bar']"]
    assert NC(instance=["foo", "bar"], regex=True).basic_exprs() == ["n.instance =~ '(foo)|(bar)'"]

    assert NC(type="foo").basic_exprs() == ["n.type = 'foo'"]
    assert NC(type="foo", regex=True).basic_exprs() == ["n.type =~ 'foo'"]
    assert NC(type=["foo", "bar"]).basic_exprs() == ["n.type in ['foo', 'bar']"]
    assert NC(type=["foo", "bar"], regex=True).basic_exprs() == ["n.type =~ '(foo)|(bar)'"]

    assert NC(status="foo").basic_exprs() == ["n.status = 'foo'"]
    assert NC(status="foo", regex=True).basic_exprs() == ["n.status = 'foo'"]  # not regex (status doesn't use regex)
    assert NC(status=["foo", "bar"]).basic_exprs() == ["n.status in ['foo', 'bar']"]
    assert NC(status=["foo", "bar"], regex=True).basic_exprs() == ["n.status in ['foo', 'bar']"]

    # Check that quotes and backslashes are escaped correctly.
    assert NC(status="f\\oo").basic_exprs() == ["n.status = 'f\\\\oo'"]
    assert NC(instance="fo'o").basic_exprs() == ["n.instance = \"fo'o\""]
    assert NC(instance="fo'o", regex=True).basic_exprs() == ["n.instance =~ \"fo'o\""]
    assert NC(instance=["fo'o", 'ba\"r']).basic_exprs() == ["n.instance in [\"fo'o\", 'ba\"r']"]
    assert NC(instance=["fo'o", 'ba"r'], regex=True).basic_exprs() == ["n.instance =~ '(fo\\'o)|(ba\"r)'"]

    assert NC(cropped=True).basic_exprs() == ["n.cropped"]
    assert NC(cropped=False).basic_exprs() == ["(NOT n.cropped OR NOT exists(n.cropped))"]

    assert NC(somaLocation=NotNull).basic_exprs() == ["exists(n.somaLocation)"]
    assert NC(somaLocation=IsNull).basic_exprs() == ["NOT exists(n.somaLocation)"]

    assert NC(inputRois=['SMP(R)', 'FB'], outputRois=['FB', 'SIP(R)'], roi_req='all').basic_exprs() == ['(n.FB AND n.`SIP(R)` AND n.`SMP(R)`)']
    assert NC(inputRois=['SMP(R)', 'FB'], outputRois=['FB', 'SIP(R)'], roi_req='any').basic_exprs() == ['(n.FB OR n.`SIP(R)` OR n.`SMP(R)`)']

    assert NC(min_pre=5).basic_exprs() == ["n.pre >= 5"]
    assert NC(min_post=5).basic_exprs() == ["n.post >= 5"]

    assert NC(bodyId=np.arange(1,6)).basic_exprs() == ["n.bodyId in n_search_bodyId"]

    ##
    ## basic_conditions()
    ##
    assert NC().basic_conditions() == ""
    assert NC().all_conditions() == ""
    assert NC.combined_conditions([NC(), NC(), NC()]) == ""

    # If 3 or fewer items are supplied, then they are used inline within the WHERE clause.
    bodies = [1,2,3]
    assert NC(bodyId=bodies).basic_conditions(comments=False) == "n.bodyId in [1, 2, 3]"

    # If more than 3 items are specified, then the items are stored in a global variable
    # which is referred to within the WHERE clause.
    bodies = [1,2,3,4,5]
    nc = NC(bodyId=bodies)
    assert nc.global_with() == dedent(f"""\
        WITH {bodies} as n_search_bodyId""")
    assert nc.basic_conditions(comments=False) == dedent("n.bodyId in n_search_bodyId")

    statuses = ['Traced', 'Orphan']
    nc = NC(status=statuses)
    assert nc.basic_conditions(comments=False) == f"n.status in {statuses}"

    statuses = ['Traced', 'Orphan', 'Assign', 'Unimportant']
    nc = NC(status=statuses)
    assert nc.global_with() == dedent(f"""\
        WITH {statuses} as n_search_status""")
    assert nc.basic_conditions(comments=False) == "n.status in n_search_status"

    # If None is included, then exists() should be checked.
    statuses = ['Traced', 'Orphan', 'Assign', None]
    nc = NC(status=statuses)
    assert nc.global_with() == dedent("""\
        WITH ['Traced', 'Orphan', 'Assign'] as n_search_status""")
    assert nc.basic_conditions(comments=False) == dedent("n.status in n_search_status OR NOT exists(n.status)")

    types = ['aaa', 'bbb', 'ccc']
    nc = NC(type=types)
    assert nc.basic_conditions(comments=False) == f"n.type in {types}"

    types = ['aaa', 'bbb', 'ccc', 'ddd']
    nc = NC(type=types)
    assert nc.global_with() == dedent(f"""\
        WITH {types} as n_search_type""")
    assert nc.basic_conditions(comments=False) == "n.type in n_search_type"

    instances = ['aaa', 'bbb', 'ccc']
    nc = NC(instance=instances)
    assert nc.basic_conditions(comments=False) == f"n.instance in {instances}"

    instances = ['aaa', 'bbb', 'ccc', 'ddd']
    nc = NC(instance=instances)
    assert nc.global_with() == dedent(f"""\
        WITH {instances} as n_search_instance""")
    assert nc.basic_conditions(comments=False) == "n.instance in n_search_instance"

    # Special case:
    # If both type and instance are supplied, then we combine them with 'OR'
    typeinst = ['aaa', 'bbb', 'ccc']
    nc = NC(type=typeinst, instance=typeinst)
    assert nc.basic_conditions(comments=False) == f"(n.type in {typeinst} OR n.instance in {typeinst})"

    typeinst = ['aaa', 'bbb', 'ccc', 'ddd']
    nc = NC(type=typeinst, instance=typeinst)
    assert nc.basic_conditions(comments=False) == "(n.type in n_search_type OR n.instance in n_search_instance)"


def test_where_expr():
    assert where_expr('bodyId', [1], matchvar='m') == 'm.bodyId = 1'
    assert where_expr('bodyId', [1,2], matchvar='m') == 'm.bodyId in [1, 2]'
    assert where_expr('bodyId', np.array([1,2]), matchvar='m') == 'm.bodyId in [1, 2]'
    assert where_expr('bodyId', []) == ""
    assert where_expr('instance', ['foo.*'], regex=True, matchvar='m') == "m.instance =~ 'foo.*'"
    assert where_expr('instance', ['foo.*', 'bar.*', 'baz.*'], regex=True, matchvar='m') == "m.instance =~ '(foo.*)|(bar.*)|(baz.*)'"

    # We use backticks in the cypher when necessary (but not otherwise).
    assert where_expr('foo/bar', [1], matchvar='m') == 'm.`foo/bar` = 1'
    assert where_expr('foo/bar', [1,2], matchvar='m') == 'm.`foo/bar` in [1, 2]'
    assert where_expr('foo/bar', np.array([1,2]), matchvar='m') == 'm.`foo/bar` in [1, 2]'
    assert where_expr('foo/bar', []) == ""


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'neuprint.tests.test_neuroncriteria']
    pytest.main(args)
