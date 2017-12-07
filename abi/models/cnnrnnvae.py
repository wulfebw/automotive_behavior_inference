import tensorflow as tf

from abi.models.rnnvae import RNNVAE

class CNNRNNVAE(RNNVAE):

    def __init__(
            self,
            max_len,
            obs_h,
            obs_w,
            obs_c,
            act_dim,
            batch_size,
            n_percept_layers=4,
            n_percept_filters=(32,64,128,128),
            **kwargs):
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.obs_c = obs_c
        self.n_percept_layers = n_percept_layers
        self.n_percept_filters = n_percept_filters
        obs_dim = 0 # not used
        super(CNNRNNVAE, self).__init__(max_len, obs_dim, act_dim, batch_size, **kwargs)

    def _build_perception(self):
        self.obs = tf.placeholder(
            tf.float32, 
            (self.batch_size, self.max_len, self.obs_h, self.obs_w, self.obs_c), 
            'obs'
        )

        hidden = tf.reshape(self.obs, (self.batch_size * self.max_len, self.obs_h, self.obs_w, self.obs_c))
        for i in range(self.n_percept_layers):
            hidden = tf.layers.conv2d(
                hidden,
                filters=self.n_percept_filters[i],
                kernel_size=3,
                strides=(2,2),
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.1))
            hidden = tf.nn.dropout(hidden, self.dropout_keep_prob_ph)   

        self.dec_inputs = tf.reshape(hidden, (self.batch_size, self.max_len, -1))
        self.enc_inputs = tf.concat((self.dec_inputs, self.act), axis=-1)
