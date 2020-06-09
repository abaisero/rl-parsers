import unittest

import numpy as np

from rl_parsers import FSC_Parser


class FSC_Test(unittest.TestCase):
    def test_tiger_optimal(self):
        parser = FSC_Parser(optimize=False)
        fsc = parser.parse_file('tests/fsc/tiger.optimal.fsc')

        self.assertListEqual(fsc.nodes, list(range(5)))
        self.assertListEqual(fsc.actions, ['listen', 'open-left', 'open-right'])
        self.assertListEqual(fsc.observations, ['obs-left', 'obs-right'])

        np.testing.assert_allclose(fsc.start, np.array([0, 0, 1, 0, 0]))

        self.assertTupleEqual(fsc.A.shape, (5, 3))
        np.testing.assert_allclose(fsc.A.sum(-1), 1.0)

        self.assertTupleEqual(fsc.T.shape, (2, 5, 5))
        np.testing.assert_allclose(fsc.T.sum(-1), 1.0)

        self.assertDictEqual(fsc.flags, {})

    def test_loadunload_optimal(self):
        parser = FSC_Parser(optimize=False)
        fsc = parser.parse_file('tests/fsc/loadunload.optimal.fsc')

        self.assertListEqual(fsc.nodes, list(range(2)))
        self.assertListEqual(fsc.actions, ['right', 'left'])
        self.assertListEqual(
            fsc.observations, ['loading', 'unloading', 'travel']
        )

        np.testing.assert_allclose(fsc.start, np.array([1, 0]))

        self.assertTupleEqual(fsc.A.shape, (2, 2))
        np.testing.assert_allclose(fsc.A.sum(-1), 1.0)

        self.assertTupleEqual(fsc.T.shape, (3, 2, 2))
        np.testing.assert_allclose(fsc.T.sum(-1), 1.0)

        self.assertDictEqual(fsc.flags, {})
