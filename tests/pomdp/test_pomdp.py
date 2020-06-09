import unittest

import numpy as np

from rl_parsers import POMDP_Parser


class POMDP_Test(unittest.TestCase):
    def test_tiger(self):
        parser = POMDP_Parser(optimize=False)
        pomdp = parser.parse_file('tests/pomdp/tiger.pomdp')

        self.assertEqual(pomdp.discount, 0.95)

        self.assertListEqual(pomdp.states, ['tiger-left', 'tiger-right'])
        self.assertListEqual(
            pomdp.actions, ['listen', 'open-left', 'open-right']
        )
        self.assertListEqual(pomdp.observations, ['obs-left', 'obs-right'])

        self.assertIsNone(pomdp.start)

        self.assertTupleEqual(pomdp.T.shape, (3, 2, 2))
        np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.O.shape, (3, 2, 2))
        np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.R.shape, (3, 2, 2, 2))
        self.assertSetEqual(set(pomdp.R.ravel().tolist()), {-100, -1, 10})

        self.assertDictEqual(pomdp.flags, {'O_includes_state': False})

    def test_loadunload(self):
        parser = POMDP_Parser(optimize=False)
        pomdp = parser.parse_file('tests/pomdp/loadunload.pomdp')

        self.assertEqual(pomdp.discount, 0.95)

        self.assertListEqual(pomdp.states, list(range(10)))
        self.assertListEqual(pomdp.actions, ['right', 'left'])
        self.assertListEqual(
            pomdp.observations, ['loading', 'unloading', 'travel']
        )

        np.testing.assert_allclose(pomdp.start, np.ones(10) / 10)

        self.assertTupleEqual(pomdp.T.shape, (2, 10, 10))
        np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.O.shape, (2, 10, 3))
        np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.R.shape, (2, 10, 10, 3))
        self.assertSetEqual(set(pomdp.R.ravel().tolist()), {0, 1})

        self.assertDictEqual(pomdp.flags, {'O_includes_state': False})

    def test_heavenhell(self):
        parser = POMDP_Parser(optimize=False)
        pomdp = parser.parse_file('tests/pomdp/heavenhell.pomdp')

        self.assertEqual(pomdp.discount, 0.99)

        self.assertListEqual(pomdp.states, list(range(20)))
        self.assertListEqual(pomdp.actions, ['N', 'S', 'E', 'W'])
        self.assertListEqual(
            pomdp.observations,
            [
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
            ],
        )

        start = np.zeros(20)
        start[0] = start[10] = 0.5
        np.testing.assert_allclose(pomdp.start, start)

        self.assertTupleEqual(pomdp.T.shape, (4, 20, 20))
        np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.O.shape, (4, 20, 11))
        np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.R.shape, (4, 20, 20, 11))
        self.assertSetEqual(set(pomdp.R.ravel().tolist()), {-1, 0, 1})

        self.assertDictEqual(pomdp.flags, {'O_includes_state': False})

    def test_floatreset(self):
        parser = POMDP_Parser(optimize=False)
        pomdp = parser.parse_file('tests/pomdp/floatreset.pomdp')

        self.assertEqual(pomdp.discount, 0.99)

        self.assertListEqual(pomdp.states, list(range(5)))
        self.assertListEqual(pomdp.actions, ['f', 'r'])
        self.assertListEqual(pomdp.observations, list(range(2)))

        start = np.zeros(5)
        start[0] = 1.0
        np.testing.assert_allclose(pomdp.start, start)

        self.assertTupleEqual(pomdp.T.shape, (2, 5, 5))
        np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.O.shape, (2, 5, 5, 2))
        np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.R.shape, (2, 5, 5, 2))
        self.assertSetEqual(set(pomdp.R.ravel().tolist()), {0, 1, 2, 3, 4})

        self.assertDictEqual(pomdp.flags, {'O_includes_state': True})

    def test_ejs2(self):
        parser = POMDP_Parser(optimize=False)
        pomdp = parser.parse_file('tests/pomdp/ejs2.POMDP')

        self.assertIsNone(pomdp.discount, None)

        self.assertListEqual(pomdp.states, list(range(2)))
        self.assertListEqual(pomdp.actions, list(range(2)))
        self.assertListEqual(pomdp.observations, list(range(2)))

        self.assertIsNone(pomdp.start)

        self.assertTupleEqual(pomdp.T.shape, (2, 2, 2))
        np.testing.assert_allclose(pomdp.T.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.O.shape, (2, 2, 2))
        np.testing.assert_allclose(pomdp.O.sum(-1), 1.0)

        self.assertTupleEqual(pomdp.R.shape, (2, 2, 2, 2))
        self.assertSetEqual(set(pomdp.R.ravel().tolist()), {0, 3, 4})

        self.assertDictEqual(pomdp.flags, {'O_includes_state': False})
