import pytest
from neuprint import Client, default_client, set_default_client, SegmentCriteria as SC
from neuprint.segmentcriteria import  where_expr
from neuprint.tests import NEUPRINT_SERVER, DATASET


@pytest.fixture(scope='module')
def client():
    c = Client(NEUPRINT_SERVER, DATASET)
    set_default_client(c)
    assert default_client() is c
    return c


def test_SegmentsCriteria(client):
    assert SC().basic_conditions() == ""
    assert SC().all_conditions() == ""
    assert SC.combined_conditions([SC(), SC(), SC()]) == ""
    
    assert SC(bodyId=123).basic_exprs() == ["n.bodyId = 123"]
    assert SC('m', bodyId=123).basic_exprs() == ["m.bodyId = 123"]
    assert SC(bodyId=[123, 456]).basic_exprs() == ["n.bodyId in [123, 456]"]
    
    assert SC(instance="foo").basic_exprs() == ["n.instance = 'foo'"]
    assert SC(instance="foo", regex=True).basic_exprs() == ["n.instance =~ 'foo'"]
    assert SC(instance=["foo", "bar"]).basic_exprs() == ["n.instance in ['foo', 'bar']"]
    with pytest.raises(AssertionError):
        SC(instance=["foo", "bar"], regex=True).basic_exprs()

    assert SC(type="foo").basic_exprs() == ["n.type = 'foo'"]
    assert SC(type="foo", regex=True).basic_exprs() == ["n.type =~ 'foo'"]
    assert SC(type=["foo", "bar"]).basic_exprs() == ["n.type in ['foo', 'bar']"]
    with pytest.raises(AssertionError):
        SC(type=["foo", "bar"], regex=True).basic_exprs()

    assert SC(status="foo").basic_exprs() == ["n.status = 'foo'"]
    assert SC(status="foo", regex=True).basic_exprs() == ["n.status = 'foo'"] # not regex
    assert SC(status=["foo", "bar"]).basic_exprs() == ["n.status in ['foo', 'bar']"]
    assert SC(status=["foo", "bar"], regex=True).basic_exprs() == ["n.status in ['foo', 'bar']"]

    assert SC(cropped=True).basic_exprs() == ["n.cropped"]
    assert SC(cropped=False).basic_exprs() == ["(NOT n.cropped OR NOT exists(n.cropped))"]

    assert SC(inputRois=['EB', 'FB'], outputRois=['FB', 'PB'], roi_req='all').basic_exprs() == ['(n.`EB` AND n.`FB` AND n.`PB`)']
    assert SC(inputRois=['EB', 'FB'], outputRois=['FB', 'PB'], roi_req='any').basic_exprs() == ['(n.`EB` OR n.`FB` OR n.`PB`)']

    assert SC(min_pre=5).basic_exprs() == ["n.pre >= 5"]
    assert SC(min_post=5).basic_exprs() == ["n.post >= 5"]


def test_where_expr():
    assert where_expr('bodyId', [1], matchvar='m') == 'm.bodyId = 1'
    assert where_expr('bodyId', [1,2], matchvar='m') == 'm.bodyId in [1, 2]'
    assert where_expr('bodyId', []) == ""
    assert where_expr('instance', ['foo.*'], regex=True, matchvar='m') == "m.instance =~ 'foo.*'"
