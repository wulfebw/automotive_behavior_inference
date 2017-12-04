
import numpy as np

from abi.misc.utils import compute_n_batches, compute_batch_idxs

class Dataset(object):
    
    def __init__(self, obs, act, lengths, batch_size=100, shuffle=True):
        self.obs = obs
        self.act = act
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_samples = len(obs)
        self.n_batches = compute_n_batches(self.n_samples, self.batch_size)

    def sample(self, n_samples=None):
        n_samples = self.batch_size if n_samples is None else n_samples
        idxs = np.random.randint(0, self.n_samples, n_samples)
        return dict(
            obs=self.obs[idxs],
            act=self.act[idxs],
            lengths=self.lengths[idxs]
        )
        
    def _shuffle(self):
        if self.shuffle:
            idxs = np.random.permutation(self.n_samples)
            self.obs = self.obs[idxs]
            self.act = self.act[idxs]
            self.lengths = self.lengths[idxs]
        
    def batches(self):
        self._shuffle()
        for bidx in range(self.n_batches):
            start = bidx * self.batch_size
            idxs = compute_batch_idxs(start, self.batch_size, self.n_samples)
            yield dict(
                obs=self.obs[idxs],
                act=self.act[idxs],
                lengths=self.lengths[idxs]
            )