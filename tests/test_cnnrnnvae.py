
import numpy as np
import tensorflow as tf
import unittest

from abi.misc.dataset import Dataset
from abi.models.cnnrnnvae import CNNRNNVAE

class TestCNNRNNVAE(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.reset_default_graph()

    def test_rnn_vae_simple(self):
        # build model
        max_len = 5
        obs_h = 32
        obs_w = 32
        obs_c = 1
        act_dim = 2
        z_dim = 2
        batch_size = 2
        kl_final = 0.0
        kl_initial = 0.0
        model = CNNRNNVAE(
            max_len, 
            obs_h,
            obs_w,
            obs_c,
            act_dim, 
            batch_size, 
            n_percept_layers=2,
            n_percept_filters=(16,16),
            z_dim=z_dim, 
            kl_steps=100,
            kl_final=kl_final,
            kl_initial=kl_initial
        )

        # generate some data 
        n_samples = batch_size
        obs = np.zeros((n_samples, max_len, obs_h, obs_w, obs_c))
        obs[:n_samples//2,:,:obs_h//2, :obs_w//2] = 1
        obs[n_samples//2:,:,obs_h//2:, obs_w//2:] = 1
        act = np.ones((n_samples, max_len, act_dim))
        act[:n_samples//2] *= -1
        lengths = np.array([max_len] * n_samples)
        dataset = Dataset(obs, act, lengths, batch_size, shuffle=False)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.train(dataset, n_epochs=500, verbose=True)
            info = model.reconstruct(obs, act, lengths)

        self.assertTrue(info['data_loss'] < -2)

if __name__ == '__main__':
    unittest.main()