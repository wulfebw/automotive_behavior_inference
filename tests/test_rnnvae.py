
import numpy as np
import tensorflow as tf
import unittest

from abi.misc.dataset import Dataset
from abi.models.rnnvae import RNNVAE

class TestRNNVAE(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.reset_default_graph()

    def test_rnn_vae_simple(self):
        # build model
        max_len = 5
        obs_dim = 1
        act_dim = 2
        z_dim = 2
        batch_size = 2
        kl_final = 0.0
        kl_initial = 0.0
        model = RNNVAE(
            max_len, 
            obs_dim, 
            act_dim, 
            batch_size, 
            z_dim=z_dim, 
            kl_steps=100,
            kl_final=kl_final,
            kl_initial=kl_initial
        )

        # generate some data 
        # z is the latent variable 
        # act is a deterministic function of z
        # obs is zeros
        # thus latent space has to be used
        n_samples = batch_size
        z = np.random.randn(n_samples, act_dim)
        obs = np.zeros((n_samples, max_len, obs_dim))
        act = np.ones((n_samples, max_len, act_dim)) * z.reshape(n_samples, 1, act_dim)
        lengths = np.array([max_len, 3])
        dataset = Dataset(obs, act, lengths, batch_size, shuffle=False)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.train(dataset, n_epochs=500, verbose=True)
            info = model.reconstruct(obs, act, lengths)

        self.assertTrue(info['data_loss'] < -2)

    def test_get_set_param_values(self):
        with tf.Session() as sess:
            model = RNNVAE(5, 2, 2, 10)
            sess.run(tf.global_variables_initializer())
            params = model.get_param_values()
            init_sum = np.sum([np.sum(p) for p in params])
            params = [p * 0 for p in params]
            model.set_param_values(params)
            new_params = model.get_param_values()
            new_sum = np.sum([np.sum(p) for p in new_params])
            orig_shapes = [np.shape(p) for p in params]
            new_shapes = [np.shape(p) for p in new_params]
            np.testing.assert_array_equal(orig_shapes, new_shapes)
            self.assertNotEqual(init_sum, new_sum)
            self.assertEqual(new_sum, 0.)

    @unittest.skipIf(__name__ != '__main__', 'run this test directly')
    def test_rnn_vae_plot(self):
        # build model
        max_len = 5
        obs_dim = 1
        act_dim = 2
        z_dim = 2
        batch_size = 100
        model = RNNVAE(
            max_len, 
            obs_dim, 
            act_dim, 
            batch_size, 
            z_dim=z_dim, 
            kl_steps=100
        )

        # generate some data 
        # z is the latent variable 
        # act is a deterministic function of z
        # obs is zeros
        # thus latent space has to be used
        n_samples = 1000
        z = np.random.randn(n_samples, act_dim)
        obs = np.zeros((n_samples, max_len, obs_dim))
        act = np.ones((n_samples, max_len, act_dim)) * z.reshape(n_samples, 1, act_dim)
        lengths = np.random.randint(2, max_len, n_samples)
        dataset = Dataset(obs, act, lengths, batch_size, shuffle=False)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.train(dataset, n_epochs=20, verbose=True)
            info = model.reconstruct(obs, act, lengths)

        plot = True
        if plot:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.subplot(1,2,1)
            plt.scatter(act[:,-1,0], act[:,-1,1], c=z.sum(1) / act_dim)
            plt.subplot(1,2,2)
            plt.scatter(act[:,-1,0], act[:,-1,1], c=info['mean'].sum(1) / act_dim)
            plt.show()

if __name__ == '__main__':
    unittest.main()