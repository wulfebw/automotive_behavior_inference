
import numpy as np
import unittest

from abi.misc.dataset import Dataset

class TestDataset(unittest.TestCase):

    def test_batches(self):
        obs = np.arange(3*2*1).reshape((3,2,1))
        act = np.arange(3*2*1).reshape((3,2,1))
        lengths = np.arange(3)
        dataset = Dataset(obs, act, lengths, batch_size=2, shuffle=False)
        batches = list(dataset.batches())
        np.testing.assert_array_equal(batches[0]['lengths'], lengths[:2])
        np.testing.assert_array_equal(batches[0]['obs'], obs[:2])
        np.testing.assert_array_equal(batches[0]['act'], act[:2])
        np.testing.assert_array_equal(batches[1]['lengths'][0], lengths[2])
        np.testing.assert_array_equal(batches[1]['obs'][0], obs[2])
        np.testing.assert_array_equal(batches[1]['act'][0], act[2])

if __name__ == '__main__':
    unittest.main()