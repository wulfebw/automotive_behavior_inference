
import collections
import numpy as np
import sys
import tensorflow as tf

import abi.core.rnn_utils as rnn_utils
import abi.misc.utils as utils

class RNNVAE(object):

    def __init__(
            self,
            max_len,
            obs_dim,
            act_dim,
            batch_size,
            dropout_keep_prob=1.,
            enc_hidden_dim=64,
            z_dim=64,
            dec_hidden_dim=64,
            kl_initial=0.0,
            kl_final=1.0,
            kl_steps=10000,
            kl_loss_min=.2,
            learning_rate=5e-4,
            grad_clip=1.):
        self.max_len = max_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.enc_hidden_dim = enc_hidden_dim
        self.z_dim = z_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.kl_initial = kl_initial
        self.kl_final = kl_final
        self.kl_steps = kl_steps
        self.kl_loss_min = kl_loss_min
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self._build_model()

    def _build_model(self):
        self._build_placeholders()
        self._build_encoder()
        self._build_decoder()
        self._build_loss()
        self._build_train_op()
        self._build_summary_op()

    def _build_placeholders(self):
        self.obs = tf.placeholder(tf.float32, (self.batch_size, self.max_len + 1, self.obs_dim), 'obs')
        self.act = tf.placeholder(tf.float32, (self.batch_size, self.max_len + 1, self.act_dim), 'act')
        self.inputs = tf.concat((self.obs, self.act), axis=-1)
        self.lengths = tf.placeholder(tf.int32, (self.batch_size,), 'lengths')
        self.sequence_mask = tf.sequence_mask(self.lengths, maxlen=self.max_len, dtype=tf.float32)
        self.dropout_keep_prop_ph = tf.placeholder_with_default(self.dropout_keep_prob, (), 'dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def _build_encoder(self):
        self.enc_cell_fw = rnn_utils._build_recurrent_cell(self.enc_hidden_dim, self.dropout_keep_prop_ph)
        self.enc_cell_bw = rnn_utils._build_recurrent_cell(self.enc_hidden_dim, self.dropout_keep_prop_ph)

        # inputs is assumed to be padded at the start with a <start> token or state
        # in the case of continuous values, probably just zeros
        # so for _encoding_ we ignore this padding
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            self.enc_cell_fw,
            self.enc_cell_bw,
            inputs=self.inputs[:,1:],
            sequence_length=self.lengths,
            dtype=tf.float32,
            time_major=False
        )

        # since the inputs are zero-padded, we can't just use the outputs
        # because the invalid timesteps will be zeros
        # instead we manually extract the last hidden states for each sample
        # in the batch and use those to define the posterior
        hidden_fw = self.enc_cell_fw.get_output(states[0])
        hidden_bw = self.enc_cell_bw.get_output(states[1])
        hidden = tf.concat((hidden_fw, hidden_bw), axis=1)

        # output the parameters of a diagonal gaussian from which to sample 
        # the z value
        self.z_mean = tf.contrib.layers.fully_connected(
            hidden, 
            self.z_dim, 
            activation_fn=None
        )
        self.z_logvar = tf.contrib.layers.fully_connected(
            hidden,
            self.z_dim,
            activation_fn=None
        )
        self.z_sigma = tf.exp(self.z_logvar / 2.)

        # sample z 
        noise = tf.random_normal((self.batch_size, self.z_dim), 0.0, 1.0)
        self.z = self.z_mean + self.z_sigma * noise

    def _build_decoder(self):
        self.dec_cell = rnn_utils._build_recurrent_cell(self.dec_hidden_dim, self.dropout_keep_prop_ph)

        # the initial state of the rnn cells is a function of z as well
        # tanh because we want the values to be zero mean centered and small
        self.initial_state = tf.nn.tanh(tf.contrib.layers.fully_connected(
            self.z,
            self.dec_cell.input_size * 2,
            activation_fn=None
        ))

        # again, note here how we ignore the last element when providing the 
        # obs seq to the _decoder_. It's because the first element is a <start>
        # token, and we want to reproduce the outputs one step ahead.
        # also note that before we passed in (obs,act), but now we are passing 
        # in just the observation with the hope of reproducing only the action
        # this will (hopefully) cause the latent z representation to contain 
        # information about common types of action sequences
        outputs, states = tf.nn.dynamic_rnn(
            self.dec_cell,
            inputs=self.obs[:,:-1],
            sequence_length=self.lengths,
            initial_state=self.initial_state,
            dtype=tf.float32,
            time_major=False
        )

        # map the outputs to mean and logvar of gaussian over actions
        act_sequence_mask = tf.reshape(self.sequence_mask, (self.batch_size, self.max_len, 1))
        epsilon = 1e-8
        outputs = tf.reshape(outputs, (-1, self.dec_cell.output_size))
        act_mean = tf.contrib.layers.fully_connected(
            outputs,
            self.act_dim,
            activation_fn=None
        )
        self.act_mean = tf.reshape(act_mean, (self.batch_size, self.max_len, self.act_dim))
        self.act_mean *= act_sequence_mask
        act_logvar = tf.contrib.layers.fully_connected(
            outputs,
            self.act_dim,
            activation_fn=None
        )
        self.act_logvar = tf.reshape(act_logvar, (self.batch_size, self.max_len, self.act_dim))
        self.act_logvar *= act_sequence_mask + epsilon
        self.act_sigma = tf.exp(self.act_logvar / 2.)
        self.act_sigma *= act_sequence_mask + epsilon

    def _build_loss(self):
        # kl loss
        # this measures the kl between the posterior distribution output from 
        # the encoder over z, and the prior of z which we choose as a unit gaussian
        self.kl_loss = -0.5 * tf.reduce_mean(
            (1 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar)))
        self.kl_loss = tf.maximum(self.kl_loss, self.kl_loss_min)

        # gradually increase the weight of the kl loss with a coefficient
        # we do this because there's no explicit reason why the model should 
        # use the z value (i.e., nothing in the objective encourages this)
        # so by starting with a small loss value for "using" the z (i.e, outputting
        # a posterior distribution over z quite different from a unit gaussian)
        # the network basically becomes reliant on the z value, and then due to 
        # the optimization being locally optimal and other factors the network 
        # continues to use the z values in informing outputs
        self.kl_weight = tf.train.polynomial_decay(
            self.kl_initial, 
            self.global_step, 
            self.kl_steps, 
            end_learning_rate=self.kl_final, 
            power=2.0,
            name='kl_weight'
        )

        # reconstruction loss
        # output mean and sigma of a gaussian for the actions 
        # and compute reconstruction loss as the -log prob of true values
        dist = tf.contrib.distributions.MultivariateNormalDiag(self.act_mean, self.act_sigma)
        data_loss = -dist.log_prob(self.act[:,1:])
        # can't remember how many times I've messed this part up
        # at this point, the data_loss has shape (batch_size, max_len)
        # a lot of the values of this array are invalid, though, because they 
        # correspond to padded values, so we have to mask them out 
        data_loss = self.sequence_mask * data_loss
        # then we want to average over the timesteps of each sample
        # since values are invalid, we can't just take the mean
        # we have to sum in order to ignore the zeros, and then divide by the 
        # lengths to get the correct average
        data_loss = tf.reduce_sum(data_loss, axis=1) / tf.cast(self.lengths, tf.float32)
        # then finally average over the batch
        self.data_loss = tf.reduce_mean(data_loss)

        self.loss = self.data_loss + self.kl_weight * self.kl_loss

        # summaries
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('data_loss', self.data_loss)
        tf.summary.scalar('kl_loss', self.kl_loss)
        tf.summary.scalar('kl_weight', self.kl_weight)

    def _build_train_op(self):
        self.var_list = tf.trainable_variables()
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # compute gradients, then clip them because otherwise they'll tend 
        # to explode
        grads_vars = optimizer.compute_gradients(self.loss, self.var_list)
        clipped_grads_vars = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v) 
            for (g,v) in grads_vars]
        self.train_op = optimizer.apply_gradients(clipped_grads_vars, global_step=self.global_step)

        # summaries
        tf.summary.scalar('grads_global_norm', tf.global_norm([g for (g,_) in grads_vars]))
        tf.summary.scalar('clipped_grads_global_norm', tf.global_norm([g for (g,_) in clipped_grads_vars]))
        tf.summary.scalar('vars_global_norm', tf.global_norm(self.var_list))
        tf.summary.scalar('learning_rate', self.learning_rate)

    def _build_summary_op(self):
        self.summary_op = tf.summary.merge_all()

    def _train_batch(self, batch, info, writer=None, train=True):
        outputs = [self.global_step, self.summary_op, self.data_loss, self.kl_loss]
        if train:
            outputs += [self.train_op]
        feed = {
            self.obs: batch['obs'],
            self.act: batch['act'],
            self.lengths: batch['lengths'],
            self.dropout_keep_prop_ph: self.dropout_keep_prob if train else 1.
        }
        sess = tf.get_default_session()
        fetched = sess.run(outputs, feed_dict=feed)
        if train:
            step, summary, data_loss, kl_loss, _ = fetched
        else:
            step, summary, data_loss, kl_loss = fetched

        if writer is not None:
            writer.add_summary(summary, step)

        info['data_loss'] += data_loss
        info['kl_loss'] += kl_loss
        info['itr'] += 1

    def _report(self, info, name, epoch, n_epochs, batch, n_batches):
        msg = '\r{} epoch: {} / {} batch: {} / {}'.format(
            name, epoch+1, n_epochs, batch+1, n_batches)
        keys = sorted(info.keys())
        for k in keys:
            if k != 'itr':
                msg += ' {}: {:.5f} '.format(k, info[k] / info['itr'])
        sys.stdout.write(msg)

    def train(
            self,
            dataset,
            val_dataset=None,
            n_epochs=100, 
            writer=None, 
            val_writer=None,
            verbose=True):
        
        for epoch in range(n_epochs):
            train_info = collections.defaultdict(float)
            for bidx, batch in enumerate(dataset.batches()):
                self._train_batch(batch, train_info, writer)
                self._report(train_info, 'train', epoch, n_epochs, bidx, dataset.n_batches)

            if val_dataset is not None:
                val_info = collections.defaultdict(float)
                for bidx, batch in enumerate(val_dataset.batches()):
                    self._train_batch(batch, val_info, val_writer, train=False)
                    self._report(val_info, 'val', epoch, n_epochs, bidx, val_dataset.n_batches)

    def reconstruct(self, obs, act, lengths):
        # setup 
        sess = tf.get_default_session()
        bs = self.batch_size
        n_samples = len(obs)
        n_batches = utils.compute_n_batches(n_samples, bs)
        
        # allocate return containers
        z = np.zeros((n_samples, self.z_dim))
        mean = np.zeros((n_samples, self.z_dim))
        sigma = np.zeros((n_samples, self.z_dim))
        act_mean = np.zeros((n_samples, self.max_len, self.act_dim))
        act_sigma = np.zeros((n_samples, self.max_len, self.act_dim))
        data_loss = 0
        kl_loss = 0

        # formulate outputs
        outputs = [
            self.z, 
            self.z_mean, 
            self.z_sigma,
            self.act_mean,
            self.act_sigma,
            self.data_loss,
            self.kl_loss
        ]

        # run the batches
        for bidx in range(n_batches):
            idxs = utils.compute_batch_idxs(bidx * bs, bs, n_samples)
            feed = {
                self.obs: obs[idxs],
                self.act: act[idxs],
                self.lengths: lengths[idxs]
            }
            fetched = sess.run(outputs, feed_dict=feed)

            # unpack
            z[idxs] = fetched[0]
            mean[idxs] = fetched[1]
            sigma[idxs] = fetched[2]
            act_mean[idxs] = fetched[3]
            act_sigma[idxs] = fetched[4]
            data_loss += fetched[5]
            kl_loss += fetched[6]

        # return the relevant info
        return dict(
            z=z, 
            mean=mean, 
            sigma=sigma,
            act_mean=act_mean,
            act_sigma=act_sigma,
            data_loss=data_loss,
            kl_loss=kl_loss
        )
