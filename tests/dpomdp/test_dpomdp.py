import unittest

import numpy as np
from rl_parsers import DPOMDP_Parser


class DPOMDP_Test(unittest.TestCase):
    def test_dectiger(self):
        parser = DPOMDP_Parser(optimize=False)
        dpomdp = parser.parse_file('tests/dpomdp/dectiger.dpomdp')

        self.assertEqual(dpomdp.discount, 1.0)

        self.assertListEqual(dpomdp.agents, [0, 1])
        self.assertListEqual(dpomdp.states, ['tiger-left', 'tiger-right'])
        self.assertListEqual(
            dpomdp.actions,
            [
                ['listen', 'open-left', 'open-right'],
                ['listen', 'open-left', 'open-right'],
            ],
        )
        self.assertListEqual(
            dpomdp.observations,
            [['hear-left', 'hear-right'], ['hear-left', 'hear-right']],
        )

        np.testing.assert_allclose(dpomdp.start, np.ones(2) / 2)

        self.assertTupleEqual(dpomdp.T.shape, (3, 3, 2, 2))
        np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(dpomdp.O.shape, (3, 3, 2, 2, 2))
        np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

        self.assertTupleEqual(dpomdp.R.shape, (3, 3, 2, 2, 2, 2))
        self.assertSetEqual(
            set(dpomdp.R.ravel().tolist()), {-101, -100, -50, -2, 9, 20}
        )

    def test_2generals(self):
        parser = DPOMDP_Parser(optimize=False)
        dpomdp = parser.parse_file('tests/dpomdp/2generals.dpomdp')

        self.assertEqual(dpomdp.discount, 1.0)

        self.assertListEqual(dpomdp.agents, [0, 1])
        self.assertListEqual(dpomdp.states, ['s_small', 's_large'])
        self.assertListEqual(
            dpomdp.actions, [['observe', 'attack'], ['observe', 'attack']],
        )
        self.assertListEqual(
            dpomdp.observations,
            [['o_small', 'o_large'], ['o_small', 'o_large']],
        )

        np.testing.assert_allclose(dpomdp.start, np.ones(2) / 2)

        self.assertTupleEqual(dpomdp.T.shape, (2, 2, 2, 2))
        np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(dpomdp.O.shape, (2, 2, 2, 2, 2))
        np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

        self.assertTupleEqual(dpomdp.R.shape, (2, 2, 2, 2, 2, 2))
        self.assertSetEqual(set(dpomdp.R.ravel().tolist()), {-20, -10, -1, 5})

    def test_recycling(self):
        parser = DPOMDP_Parser(optimize=False)
        dpomdp = parser.parse_file('tests/dpomdp/recycling.dpomdp')

        self.assertEqual(dpomdp.discount, 0.9)

        self.assertListEqual(dpomdp.agents, [0, 1])
        self.assertListEqual(dpomdp.states, [0, 1, 2, 3])
        self.assertListEqual(
            dpomdp.actions,
            [
                ['searchbig', 'searchlittle', 'waitandrecharge'],
                ['searchbig', 'searchlittle', 'waitandrecharge'],
            ],
        )
        self.assertListEqual(dpomdp.observations, [[0, 1], [0, 1]])

        np.testing.assert_allclose(dpomdp.start, np.array([1.0, 0.0, 0.0, 0.0]))

        self.assertTupleEqual(dpomdp.T.shape, (3, 3, 4, 4))
        np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(dpomdp.O.shape, (3, 3, 4, 2, 2))
        np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

        self.assertTupleEqual(dpomdp.R.shape, (3, 3, 4, 4, 2, 2))
        self.assertSetEqual(
            set(dpomdp.R.ravel().tolist()),
            {
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
            },
        )

    def test_prisoners(self):
        parser = DPOMDP_Parser(optimize=False)
        dpomdp = parser.parse_file('tests/dpomdp/prisoners.dpomdp')

        self.assertEqual(dpomdp.discount, 1.0)

        self.assertListEqual(dpomdp.agents, [0, 1])
        self.assertListEqual(dpomdp.states, ['NULL_STATE'])
        self.assertListEqual(
            dpomdp.actions,
            [['StaySilent', 'Betray'], ['StaySilent', 'Betray']],
        )
        self.assertListEqual(
            dpomdp.observations,
            [['O_StaySilent', 'O_Betray'], ['O_StaySilent', 'O_Betray']],
        )

        np.testing.assert_allclose(dpomdp.start, np.ones(1))

        self.assertTupleEqual(dpomdp.T.shape, (2, 2, 1, 1))
        np.testing.assert_allclose(dpomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(dpomdp.O.shape, (2, 2, 1, 2, 2))
        np.testing.assert_allclose(dpomdp.O.sum((-2, -1)), 1.0)

        self.assertTupleEqual(dpomdp.R.shape, (2, 2, 1, 1, 2, 2))
        self.assertSetEqual(
            set(dpomdp.R.ravel().tolist()), {-10.0, -5.0, -1.0, 0.0},
        )
