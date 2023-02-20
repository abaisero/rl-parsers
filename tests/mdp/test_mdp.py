import numpy as np

from rl_parsers import MDP_Parser


def test_gridworld():
    parser = MDP_Parser(optimize=False)
    mdp = parser.parse_file('tests/mdp/gridworld.mdp')

    assert mdp.discount == 0.95

    assert mdp.states == list(range(11))
    assert mdp.actions == ['north', 'south', 'east', 'west']

    assert mdp.start is None

    assert mdp.T.shape == (4, 11, 11)
    np.testing.assert_allclose(mdp.T.sum(-1), 1.0)

    assert mdp.R.shape == (4, 11, 11)
    assert set(mdp.R.flat) == {-0.1, -1, 1}

    assert mdp.flags == {}
