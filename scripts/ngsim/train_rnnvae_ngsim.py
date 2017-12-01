import h5py
import numpy as np
import tensorflow as tf

from abi.misc.dataset import Dataset
from abi.misc.utils import load_data
from abi.models.rnnvae import RNNVAE

if __name__ == '__main__':
    obs_keys = [
        'relative_offset', 'relative_heading', 'velocity', 'length', 'width',
        'lane_curvature', 'markerdist_left', 'markerdist_right', 'jerk',
        'angular_rate_frenet', 'timegap', 'time_to_collision',
        'is_colliding', 'out_of_lane', 'negative_velocity'
    ]
    nbeams = 20
    obs_keys += ['lidar_{}'.format(i) for i in range(1, nbeams+1)]
    obs_keys += ['rangerate_lidar_{}'.format(i) for i in range(1, nbeams+1)]
    act_keys = ['accel', 'turn_rate_frenet']
    data = load_data(
        filepath='../../data/trajectories/ngsim.h5',
        debug_size=None,
        mode='ngsim',
        min_length=5,
        obs_keys=obs_keys,
        act_keys=act_keys,
        load_y=False
    )
    obs = data['obs']
    act = data['act']
    lengths = data['lengths']
    obs_keys = data['obs_keys']
    act_keys = data['act_keys']
    max_len = data['max_len']
    obs_dim = data['obs_dim']
    act_dim = data['act_dim']
    val_obs, val_act, val_lengths, val_y = data['val_obs'], data['val_act'], data['val_lengths'], data['val_y']

    batch_size = 100
    dataset = Dataset(
        np.copy(obs), 
        np.copy(act), 
        np.copy(lengths), 
        batch_size, 
        shuffle=True
    )

    z_dim = 2
    kl_final = 1.

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    model = RNNVAE(
        max_len, 
        obs_dim, 
        act_dim, 
        batch_size, 
        z_dim=z_dim, 
        enc_hidden_dim=32,
        dec_hidden_dim=32,
        kl_steps=1000,
        kl_final=kl_final,
        learning_rate=1e-3
    )
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('../../data/summaries/ngsim')
    model.train(
        dataset, 
        writer=writer,
        n_epochs=1000, 
        verbose=True
    )

