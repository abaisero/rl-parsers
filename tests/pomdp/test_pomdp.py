import numpy as np

from rl_parsers import POMDP_Parser


def test_tiger():
    parser = POMDP_Parser(optimize=False)
    pomdp = parser.parse_file('tests/pomdp/tiger.pomdp')

    assert pomdp.discount == 0.95

    assert pomdp.states == ['tiger-left', 'tiger-right']
    assert pomdp.actions == ['listen', 'open-left', 'open-right']
    assert pomdp.observations == ['obs-left', 'obs-right']

    assert pomdp.start is None

    assert pomdp.T.shape == (3, 2, 2)
    np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

    assert pomdp.O.shape == (3, 2, 2)
    np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

    assert pomdp.R.shape == (3, 2, 2, 2)
    assert set(pomdp.R.flat) == {-100, -1, 10}

    assert pomdp.flags == {'O_includes_state': False}


def test_loadunload():
    parser = POMDP_Parser(optimize=False)
    pomdp = parser.parse_file('tests/pomdp/loadunload.pomdp')

    assert pomdp.discount == 0.95

    assert pomdp.states == list(range(10))
    assert pomdp.actions == ['right', 'left']
    assert pomdp.observations == ['loading', 'unloading', 'travel']

    np.testing.assert_allclose(pomdp.start, np.full(10, 0.1))

    assert pomdp.T.shape == (2, 10, 10)
    np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

    assert pomdp.O.shape == (2, 10, 3)
    np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

    assert pomdp.R.shape == (2, 10, 10, 3)
    assert set(pomdp.R.flat) == {0, 1}

    assert pomdp.flags == {'O_includes_state': False}


def test_heavenhell():
    parser = POMDP_Parser(optimize=False)
    pomdp = parser.parse_file('tests/pomdp/heavenhell.pomdp')

    assert pomdp.discount == 0.99

    assert pomdp.states == list(range(20))
    assert pomdp.actions == ['N', 'S', 'E', 'W']
    assert pomdp.observations == [
        's0',
        's1',
        's2',
        's3',
        's4',
        's5',
        's6',
        's7',
        's8',
        'left',
        'right',
    ]

    start = np.zeros(20)
    start[0] = start[10] = 0.5
    np.testing.assert_allclose(pomdp.start, start)

    assert pomdp.T.shape == (4, 20, 20)
    np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

    assert pomdp.O.shape == (4, 20, 11)
    np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

    assert pomdp.R.shape == (4, 20, 20, 11)
    assert set(pomdp.R.flat) == {-1, 0, 1}

    assert pomdp.flags == {'O_includes_state': False}


def test_floatreset():
    parser = POMDP_Parser(optimize=False)
    pomdp = parser.parse_file('tests/pomdp/floatreset.pomdp')

    assert pomdp.discount == 0.99

    assert pomdp.states == list(range(5))
    assert pomdp.actions == ['f', 'r']
    assert pomdp.observations == list(range(2))

    start = np.zeros(5)
    start[0] = 1.0
    np.testing.assert_allclose(pomdp.start, start)

    assert pomdp.T.shape == (2, 5, 5)
    np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

    assert pomdp.O.shape == (2, 5, 5, 2)
    np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

    assert pomdp.R.shape == (2, 5, 5, 2)
    assert set(pomdp.R.flat) == {0, 1, 2, 3, 4}

    assert pomdp.flags == {'O_includes_state': True}


def test_ejs2():
    parser = POMDP_Parser(optimize=False)
    pomdp = parser.parse_file('tests/pomdp/ejs2.POMDP')

    assert pomdp.discount == None

    assert pomdp.states == list(range(2))
    assert pomdp.actions == list(range(2))
    assert pomdp.observations == list(range(2))

    assert pomdp.start is None

    assert pomdp.T.shape == (2, 2, 2)
    np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

    assert pomdp.O.shape == (2, 2, 2)
    np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

    assert pomdp.R.shape == (2, 2, 2, 2)
    assert set(pomdp.R.flat) == {0, 3, 4}

    assert pomdp.flags == {'O_includes_state': False}
