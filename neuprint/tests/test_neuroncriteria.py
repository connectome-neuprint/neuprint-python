import pytest
from neuprint import Client, default_client, set_default_client, NeuronCriteria as NC
from neuprint.neuroncriteria import  where_expr
from neuprint.tests import NEUPRINT_SERVER, DATASET


@pytest.fixture(scope='module')
def client():
    c = Client(NEUPRINT_SERVER, DATASET)
    set_default_client(c)
    assert default_client() is c
    return c


def test_NeuronCriteria(client):
    assert NC().basic_conditions() == ""
    assert NC().all_conditions() == ""
    assert NC.combined_conditions([NC(), NC(), NC()]) == ""
    
    assert NC(bodyId=123).basic_exprs() == ["n.bodyId = 123"]
    assert NC('m', bodyId=123).basic_exprs() == ["m.bodyId = 123"]
    assert NC(bodyId=[123, 456]).basic_exprs() == ["n.bodyId in [123, 456]"]
    
    assert NC(instance="foo").basic_exprs() == ["n.instance = 'foo'"]
    assert NC(instance="foo", regex=True).basic_exprs() == ["n.instance =~ 'foo'"]
    assert NC(instance=["foo", "bar"]).basic_exprs() == ["n.instance in ['foo', 'bar']"]
    with pytest.raises(AssertionError):
        NC(instance=["foo", "bar"], regex=True).basic_exprs()

    assert NC(type="foo").basic_exprs() == ["n.type = 'foo'"]
    assert NC(type="foo", regex=True).basic_exprs() == ["n.type =~ 'foo'"]
    assert NC(type=["foo", "bar"]).basic_exprs() == ["n.type in ['foo', 'bar']"]
    with pytest.raises(AssertionError):
        NC(type=["foo", "bar"], regex=True).basic_exprs()

    assert NC(status="foo").basic_exprs() == ["n.status = 'foo'"]
    assert NC(status="foo", regex=True).basic_exprs() == ["n.status = 'foo'"] # not regex
    assert NC(status=["foo", "bar"]).basic_exprs() == ["n.status in ['foo', 'bar']"]
    assert NC(status=["foo", "bar"], regex=True).basic_exprs() == ["n.status in ['foo', 'bar']"]

    assert NC(cropped=True).basic_exprs() == ["n.cropped"]
    assert NC(cropped=False).basic_exprs() == ["(NOT n.cropped OR NOT exists(n.cropped))"]

    assert NC(inputRois=['EB', 'FB'], outputRois=['FB', 'PB'], roi_req='all').basic_exprs() == ['(n.`EB` AND n.`FB` AND n.`PB`)']
    assert NC(inputRois=['EB', 'FB'], outputRois=['FB', 'PB'], roi_req='any').basic_exprs() == ['(n.`EB` OR n.`FB` OR n.`PB`)']

    assert NC(min_pre=5).basic_exprs() == ["n.pre >= 5"]
    assert NC(min_post=5).basic_exprs() == ["n.post >= 5"]


def test_where_expr():
    assert where_expr('bodyId', [1], matchvar='m') == 'm.bodyId = 1'
    assert where_expr('bodyId', [1,2], matchvar='m') == 'm.bodyId in [1, 2]'
    assert where_expr('bodyId', []) == ""
    assert where_expr('instance', ['foo.*'], regex=True, matchvar='m') == "m.instance =~ 'foo.*'"
