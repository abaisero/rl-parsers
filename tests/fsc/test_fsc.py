import numpy as np

from rl_parsers import FSC_Parser


def test_tiger_optimal():
    parser = FSC_Parser(optimize=False)
    fsc = parser.parse_file('tests/fsc/tiger.optimal.fsc')

    assert fsc.nodes == list(range(5))
    assert fsc.actions == ['listen', 'open-left', 'open-right']
    assert fsc.observations == ['obs-left', 'obs-right']

    np.testing.assert_allclose(fsc.start, np.array([0, 0, 1, 0, 0]))

    assert fsc.A.shape == (5, 3)
    np.testing.assert_allclose(fsc.A.sum(-1), 1.0)

    assert fsc.T.shape == (2, 5, 5)
    np.testing.assert_allclose(fsc.T.sum(-1), 1.0)

    assert fsc.flags == {}


def test_loadunload_optimal():
    parser = FSC_Parser(optimize=False)
    fsc = parser.parse_file('tests/fsc/loadunload.optimal.fsc')

    assert fsc.nodes == list(range(2))
    assert fsc.actions == ['right', 'left']
    assert fsc.observations == ['loading', 'unloading', 'travel']

    np.testing.assert_allclose(fsc.start, np.array([1, 0]))

    assert fsc.A.shape == (2, 2)
    np.testing.assert_allclose(fsc.A.sum(-1), 1.0)

    assert fsc.T.shape == (3, 2, 2)
    np.testing.assert_allclose(fsc.T.sum(-1), 1.0)

    assert fsc.flags == {}
