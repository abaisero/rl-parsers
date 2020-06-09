import unittest

import numpy as np

from rl_parsers import MDP_Parser


class MDP_Test(unittest.TestCase):
    def test_gridworld(self):
        parser = MDP_Parser(optimize=False)
        mdp = parser.parse_file('tests/mdp/gridworld.mdp')

        self.assertEqual(mdp.discount, 0.95)

        self.assertListEqual(mdp.states, list(range(11)))
        self.assertListEqual(mdp.actions, ['north', 'south', 'east', 'west'])

        self.assertIsNone(mdp.start)

        self.assertTupleEqual(mdp.T.shape, (4, 11, 11))
        np.testing.assert_allclose(mdp.T.sum(-1), 1.0)

        self.assertTupleEqual(mdp.R.shape, (4, 11, 11))
        self.assertSetEqual(set(mdp.R.ravel().tolist()), {-0.1, -1, 1})

        self.assertDictEqual(mdp.flags, {})
