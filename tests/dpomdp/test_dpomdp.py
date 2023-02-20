import numpy as np

from rl_parsers import DPOMDP_Parser


def test_dectiger():
    parser = DPOMDP_Parser(optimize=False)
    dpomdp = parser.parse_file('tests/dpomdp/dectiger.dpomdp')

    assert dpomdp.discount == 1.0

    assert dpomdp.agents == [0, 1]
    assert dpomdp.states == ['tiger-left', 'tiger-right']
    assert dpomdp.actions == [
        ['listen', 'open-left', 'open-right'],
        ['listen', 'open-left', 'open-right'],
    ]
    assert dpomdp.observations == [
        ['hear-left', 'hear-right'],
        ['hear-left', 'hear-right'],
    ]

    np.testing.assert_allclose(dpomdp.start, np.ones(2) / 2)

    assert dpomdp.T.shape == (3, 3, 2, 2)
    np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

    assert dpomdp.O.shape == (3, 3, 2, 2, 2)
    np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

    assert dpomdp.R.shape == (3, 3, 2, 2, 2, 2)
    assert set(dpomdp.R.flat) == {-101, -100, -50, -2, 9, 20}


def test_2generals():
    parser = DPOMDP_Parser(optimize=False)
    dpomdp = parser.parse_file('tests/dpomdp/2generals.dpomdp')

    assert dpomdp.discount == 1.0

    assert dpomdp.agents == [0, 1]
    assert dpomdp.states == ['s_small', 's_large']
    assert dpomdp.actions == [['observe', 'attack'], ['observe', 'attack']]
    assert dpomdp.observations == [['o_small', 'o_large'], ['o_small', 'o_large']]

    np.testing.assert_allclose(dpomdp.start, np.ones(2) / 2)

    assert dpomdp.T.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

    assert dpomdp.O.shape == (2, 2, 2, 2, 2)
    np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

    assert dpomdp.R.shape == (2, 2, 2, 2, 2, 2)
    assert set(dpomdp.R.flat) == {-20, -10, -1, 5}


def test_recycling():
    parser = DPOMDP_Parser(optimize=False)
    dpomdp = parser.parse_file('tests/dpomdp/recycling.dpomdp')

    assert dpomdp.discount == 0.9

    assert dpomdp.agents == [0, 1]
    assert dpomdp.states == [0, 1, 2, 3]
    assert dpomdp.actions == [
        ['searchbig', 'searchlittle', 'waitandrecharge'],
        ['searchbig', 'searchlittle', 'waitandrecharge'],
    ]
    assert dpomdp.observations == [[0, 1], [0, 1]]

    np.testing.assert_allclose(dpomdp.start, np.array([1.0, 0.0, 0.0, 0.0]))

    assert dpomdp.T.shape == (3, 3, 4, 4)
    np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

    assert dpomdp.O.shape == (3, 3, 4, 2, 2)
    np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

    assert dpomdp.R.shape == (3, 3, 4, 4, 2, 2)
    assert set(dpomdp.R.flat) == {
        -0.4,
        -1.44,
        -1.6,
        -3.0,
        -3.55,
        -3.88,
        0.0,
        0.5,
        1.2,
        2.0,
        4.0,
        5.0,
    }


def test_prisoners():
    parser = DPOMDP_Parser(optimize=False)
    dpomdp = parser.parse_file('tests/dpomdp/prisoners.dpomdp')

    assert dpomdp.discount == 1.0

    assert dpomdp.agents == [0, 1]
    assert dpomdp.states == ['NULL_STATE']
    assert dpomdp.actions == [['StaySilent', 'Betray'], ['StaySilent', 'Betray']]
    assert dpomdp.observations == [
        ['O_StaySilent', 'O_Betray'],
        ['O_StaySilent', 'O_Betray'],
    ]

    np.testing.assert_allclose(dpomdp.start, np.ones(1))

    assert dpomdp.T.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

    assert dpomdp.O.shape == (2, 2, 1, 2, 2)
    np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

    assert dpomdp.R.shape == (2, 2, 1, 1, 2, 2)
    assert set(dpomdp.R.flat) == {-10.0, -5.0, -1.0, 0.0}
