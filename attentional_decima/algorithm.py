import os

import numpy as np
from tf_compat import tf

from params import args
from utils.tf_utils import build_mlp, huber_loss


class AttentionalDecima():

    def __init__(self, save_path='dummy', act_fn=tf.nn.tanh):
        self._save_path = os.path.join(save_path, 'model-ckpt')
        self._act_fn = act_fn

        self._node_input_dim = args.node_input_dim
        self._job_input_dim = args.job_input_dim
        self._hid_dims = args.hid_dims
        self._output_dim = args.output_dim
        self._max_depth = args.max_depth
        self._gcn_num_heads = args.num_heads
        self._eps = args.eps
        self._exec_cap = args.exec_cap
        self._value_weight = args.value_weight
        self._weight_decay_rate = args.weight_decay_rate
        self._grads_norm_clip = args.grads_norm_clip
        self._is_clip_ratio = args.is_clip_ratio
        self._value_clip = args.value_clip
        self._num_saved_models = args.num_saved_models
        self._saved_model = args.saved_model

        self.__init()

    def __init(self):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        with tf.name_scope('placeholders'):
            self._add_placeholders_op()
        with tf.variable_scope('policy'):
            self._build_policy_op()
        with tf.name_scope('vars'):
            self._build_get_vars_op()
            self._build_set_vars_op()
        with tf.name_scope('loss'):
            self._add_loss_op()
        with tf.name_scope('train'):
            self._build_train_op()

        # network paramter saver
        self._saver = tf.train.Saver(max_to_keep=self._num_saved_models)
        self._sess.run(tf.initializers.global_variables())

        if self._saved_model is not None:
            self._saver.restore(self._sess, self._saved_model)

    def _add_placeholders_op(self):
        # (batch_size, totoal_num_nodes, node_input_dim)
        self._node_inputs_ph = tf.placeholder(tf.float32, [None, None, self._node_input_dim], 'node_inputs')
        # (batch_size, num_jobs, job_input_dim)
        self._job_inputs_ph = tf.placeholder(tf.float32, [None, None, self._job_input_dim], 'job_inputs')

        self._batch_size = tf.shape(self._node_inputs_ph)[0]
        self._totoal_num_nodes = tf.shape(self._node_inputs_ph)[1]
        self._num_jobs = tf.shape(self._job_inputs_ph)[1]

        # (batch_size * totoal_num_nodes, batch_size * totoal_num_nodes)
        self._gcn_mats_phs = [
            tf.sparse_placeholder(tf.float32, [None, None], 'gcn_mats_{}'.format(i)) for i in range(self._max_depth)
        ]
        # (batch_size * totoal_num_nodes, 1)
        self._gcn_masks_phs = [
            tf.placeholder(tf.float32, [None, 1], 'gcn_masks_{}'.format(i)) for i in range(self._max_depth)
        ]

        # (batch_size, num_jobs, totoal_num_nodes)
        self._unfinished_nodes_each_job_ph = \
            tf.placeholder(tf.float32, [None, None, None], 'unfinished_nodes_each_job')

        # (batch_size, total_num_nodes, num_dags)
        # map back the dag summeraization to each node, same for each batch
        self._dag_summ_backward_map_ph = tf.placeholder(tf.float32, [None, None, None], 'dag_summ_backward_map')

        # (batch_size, totoal_num_nodes)
        self._node_valid_mask_ph = tf.placeholder(tf.float32, [None, None], 'node_valid_mask')
        # (batch_size, num_dags, exec_cap)
        self._job_valid_mask_ph = tf.placeholder(tf.float32, [None, None, self._exec_cap], 'job_valid_mask')

        # (batch_size, num_total_nodes)
        self._node_act_vec_ph = tf.placeholder(tf.float32, [None, None], 'node_act_vec')
        # (batch_size, num_dags, exec_cap)
        self._job_act_vec_ph = tf.placeholder(tf.float32, [None, None, self._exec_cap], 'job_act_vec')
        # (batch_size, 1)
        self._old_act_logp_ph = tf.placeholder(tf.float32, [None, 1], 'old_act_logp')
        # (batch_size, 1)
        self._rtg_ph = tf.placeholder(tf.float32, [None, 1], 'reward_to_go')
        self._old_value_ph = tf.placeholder(tf.float32, [None, 1], 'old_value')

        self._ent_weight = tf.placeholder(tf.float32, [], 'entropy_weight')
        self._lr = tf.placeholder(tf.float32,  [], 'learning_rate')

    def _build_gcn_op(self):
        with tf.variable_scope('gcn', reuse=tf.AUTO_REUSE):
            outs = []

            for i in range(self._gcn_num_heads):
                seq_v = tf.layers.conv1d(self._node_inputs_ph, self._output_dim, 1, use_bias=False, name='seq_v_{}'.format(i))
                bias = tf.get_variable(
                    'head_{}_bias'.format(i),
                    [self._output_dim,],
                    initializer=tf.initializers.zeros
                )

                for depth in range(self._max_depth):
                    f_1 = tf.layers.conv1d(seq_v, 1, 1, name='head_{}_1'.format(i))
                    f_2 = tf.layers.conv1d(seq_v, 1, 1, name='head_{}_2'.format(i))

                    # (batch_size * totoal_num_nodes, 1)
                    f_1 = tf.reshape(f_1, [-1, 1])
                    f_2 = tf.reshape(f_2, [-1, 1])

                    # (batch_size * totoal_num_nodes, batch_size * totoal_num_nodes)
                    f_1 = self._gcn_mats_phs[depth] * f_1
                    f_2 = self._gcn_mats_phs[depth] * tf.transpose(f_2, [1, 0])

                    logits = tf.sparse_add(f_1, f_2)
                    lrelu = tf.SparseTensor(
                        indices=logits.indices,
                        values=tf.nn.leaky_relu(logits.values),
                        dense_shape=logits.dense_shape
                    )
                    # (batch_size * totoal_num_nodes, batch_size * totoal_num_nodes)
                    coefs = tf.sparse_softmax(lrelu)

                    v_ravel = tf.reshape(seq_v, [-1, self._output_dim])
                    vals = tf.sparse_tensor_dense_matmul(coefs, v_ravel)
                    vals = vals * self._gcn_masks_phs[depth]
                    vals = tf.nn.bias_add(vals, bias)
                    vals = tf.reshape(vals, [self._batch_size, -1, self._output_dim])
                    seq_v = seq_v + vals

                outs.append(seq_v)

            self._gcn_outputs = self._act_fn(tf.add_n(outs) / self._gcn_num_heads)

    def _build_gsn_op(self):
        with tf.variable_scope('gsn'):
            gsn_inputs = tf.concat([self._node_inputs_ph, self._gcn_outputs], axis=-1)
            gsn_inputs = tf.matmul(self._unfinished_nodes_each_job_ph, gsn_inputs)

            self._dag_level_summ = build_mlp(
                gsn_inputs,
                self._hid_dims,
                self._output_dim,
                'gsn',
                self._act_fn
            )

            cell = tf.keras.layers.GRUCell(self._output_dim)
            rnn_layer = tf.keras.layers.RNN(cell)
            self._global_level_summ = rnn_layer(self._dag_level_summ)

    def _build_policy_op(self):
        self._build_gcn_op()
        self._build_gsn_op()

        with tf.variable_scope('actor'):
            gsn_dag_summ_extend_node = tf.matmul(self._dag_summ_backward_map_ph, self._dag_level_summ)
            global_level_summ = tf.expand_dims(self._global_level_summ, axis=1)
            gsn_global_summ_extend_node = tf.tile(global_level_summ, [1, self._totoal_num_nodes, 1])
            gsn_global_summ_extend_job = tf.tile(global_level_summ, [1, self._num_jobs, 1])

            merged_node = tf.concat(
                [
                    self._node_inputs_ph,
                    self._gcn_outputs,
                    gsn_dag_summ_extend_node,
                    gsn_global_summ_extend_node
                ],
                axis=-1
            )
            merged_job = tf.concat(
                [self._job_inputs_ph, self._dag_level_summ, gsn_global_summ_extend_job],
                axis=-1
            )
            expanded_job = self._expand_job_features(merged_job)
            node_outputs = build_mlp(merged_node, self._hid_dims, 1, 'node', self._act_fn)
            node_outputs = tf.reshape(node_outputs, [self._batch_size, -1])
            node_valid_mask = (self._node_valid_mask_ph - 1.0) * 1e32
            self._node_logits = node_outputs + node_valid_mask
            self._node_probs = tf.nn.softmax(self._node_logits)

            job_outputs = build_mlp(expanded_job, self._hid_dims, 1, 'job', self._act_fn)
            job_outputs = tf.reshape(job_outputs, [self._batch_size, -1, self._exec_cap])
            job_valid_mask = (self._job_valid_mask_ph - 1.0) * 1e32
            self._job_logits = job_outputs + job_valid_mask
            self._job_probs = tf.nn.softmax(self._job_logits)

            node_noise = tf.random.uniform(shape=tf.shape(self._node_logits))
            job_noise = tf.random.uniform(shape=tf.shape(self._job_logits))
            self._node_act = tf.argmax(self._node_logits - tf.log(-tf.log(node_noise)), axis=-1)
            self._job_acts = tf.argmax(self._job_logits - tf.log(-tf.log(job_noise)), axis=-1)

        with tf.variable_scope('cirtic'):
            node_valid_mask = tf.expand_dims(self._node_valid_mask_ph, axis=-1)
            merged_node = tf.concat([self._node_inputs_ph, self._gcn_outputs], axis=-1)
            merged_node = tf.reduce_sum(merged_node * node_valid_mask, axis=1)
            merged_job = tf.concat([self._job_inputs_ph, self._dag_level_summ], axis=-1)
            merged_job = tf.reduce_sum(merged_job, axis=1)
            merged_state = tf.concat([merged_node, merged_job], axis=-1)
            self._value = build_mlp(merged_state, self._hid_dims, 1, 'value', self._act_fn)

    def _add_loss_op(self):
        with tf.name_scope('reward_loss'):
            _selected_node_prob = tf.reduce_sum(
                self._node_probs * self._node_act_vec_ph,
                axis=-1,
                keepdims=True
            )
            _selected_job_prob = tf.reduce_sum(
                tf.reduce_sum(self._job_probs * self._job_act_vec_ph, axis=-1),
                axis=-1,
                keepdims=True
            )
            act_logp = tf.log(_selected_node_prob + self._eps) + tf.log(_selected_job_prob + self._eps)
            self._is_ratio = tf.exp(act_logp - self._old_act_logp_ph)
            is_ratio = tf.clip_by_value(self._is_ratio, self._eps, self._is_clip_ratio)
            is_ratio = tf.stop_gradient(is_ratio)
            self._reward_loss = -tf.reduce_sum(is_ratio * act_logp * (self._rtg_ph - self._old_value_ph))

        with tf.name_scope('entropy_loss'):
            _node_ent = -tf.reduce_sum(self._node_probs * tf.log(self._node_probs + self._eps))
            # normalize node entropy
            _node_ent = _node_ent / tf.log(tf.cast(self._totoal_num_nodes, tf.float32))

            _node_probs = tf.expand_dims(self._node_probs, axis=-1)
            _prob_of_each_job = tf.matmul(self._unfinished_nodes_each_job_ph, _node_probs)
            _prob_of_each_job = tf.squeeze(_prob_of_each_job, axis=-1)
            _pre_job_ent = -tf.reduce_sum(self._job_probs * tf.log(self._job_probs + self._eps), axis=-1)
            _job_ent = tf.reduce_sum(_prob_of_each_job * _pre_job_ent)
            # normalize job entropy
            _job_ent = _job_ent / tf.log(float(self._exec_cap))

            self._ent_loss = _node_ent + _job_ent

        with tf.name_scope('value_loss'):
            val_loss_1 = huber_loss(self._rtg_ph, self._value)
            val_pred_clipped = tf.clip_by_value(
                self._value - self._old_value_ph,
                -self._value_clip,
                self._value_clip
            ) + self._old_value_ph
            val_loss_2 = huber_loss(self._rtg_ph, self._value)
            self._val_loss = tf.reduce_sum(tf.minimum(val_loss_1, val_loss_2))

        self._all_loss = \
            self._reward_loss - self._ent_weight * self._ent_loss + self._value_weight * self._val_loss

    def _build_train_op(self):
        self._optimizer = tf.contrib.opt.AdamWOptimizer(self._weight_decay_rate, self._lr)
        grads_and_vars = self._optimizer.compute_gradients(self._all_loss)
        grads, alg_vars = zip(*grads_and_vars)
        self._grads = list(grads)
        clipped_grads, _ = tf.clip_by_global_norm(self._grads, self._grads_norm_clip)
        self._train_op = self._optimizer.apply_gradients(zip(clipped_grads, alg_vars))

    def predict(
        self,
        node_inputs,
        job_inputs,
        node_valid_mask,
        job_valid_mask,
        gcn_mats,
        gcn_masks,
        unfinished_nodes_each_job,
        dag_summ_backward_map
    ):
        gcn_mats_tf = []
        for gcn_mat in gcn_mats:
            gcn_mats_tf.append(gcn_mat.to_tf_sp_tensor_value())

        feed_dict = {
            k: v for k, v in zip(
                [
                    self._node_inputs_ph,
                    self._job_inputs_ph,
                    self._node_valid_mask_ph,
                    self._job_valid_mask_ph,
                    self._unfinished_nodes_each_job_ph,
                    self._dag_summ_backward_map_ph
                ] + self._gcn_mats_phs + self._gcn_masks_phs,
                [
                    node_inputs,
                    job_inputs,
                    node_valid_mask,
                    job_valid_mask,
                    unfinished_nodes_each_job,
                    dag_summ_backward_map
                ] + gcn_mats_tf + gcn_masks
            )
        }

        return self._sess.run(
            [self._node_act, self._job_acts, self._node_probs, self._job_probs, self._value],
            feed_dict=feed_dict
        )

    def compute_grads(self, full_batch, ent_weight):
        all_grads, all_losses, reward_losses, ent_losses, val_losses, is_ratios = [], [], [], [], [], []
        # np.random.shuffle(full_batch)
        for mini_batch in full_batch:
            grads, all_loss, reward_loss, ent_loss, val_loss, is_ratio = \
                self._get_gradients(mini_batch, ent_weight)
            all_grads.append(grads)
            all_losses.append(all_loss)
            reward_losses.append(reward_loss)
            ent_losses.append(ent_loss)
            val_losses.append(val_loss)
            is_ratios.append(is_ratio)

        ground_grads = self._aggregate_grads(all_grads)
        return (
            ground_grads,
            np.sum(all_losses),
            np.sum(reward_losses),
            np.sum(ent_losses),
            np.sum(val_losses),
            np.concatenate(is_ratios),
        )

    def _get_gradients(self, mini_batch, ent_weight):
        gcn_mats = []
        for gcn_mat in mini_batch['gcn_mats']:
            gcn_mats.append(tf.SparseTensorValue(*gcn_mat))
        feed_dict = {
            k: v for k, v in zip(
                [
                    self._node_inputs_ph,
                    self._job_inputs_ph,
                    self._node_valid_mask_ph,
                    self._job_valid_mask_ph,
                    self._unfinished_nodes_each_job_ph,
                    self._dag_summ_backward_map_ph,
                    self._node_act_vec_ph,
                    self._job_act_vec_ph,
                    self._old_act_logp_ph,
                    self._rtg_ph,
                    self._old_value_ph,
                    self._ent_weight
                ] + self._gcn_mats_phs + self._gcn_masks_phs,
                [
                    mini_batch['node_inputs'],
                    mini_batch['job_inputs'],
                    mini_batch['node_valid_mask'],
                    mini_batch['job_valid_mask'],
                    mini_batch['unfinished_nodes_each_job'],
                    mini_batch['dag_summ_backward_map'],
                    mini_batch['node_act_vec'],
                    mini_batch['job_act_vec'],
                    mini_batch['old_act_logp'],
                    mini_batch['reward_to_go'],
                    mini_batch['old_value'],
                    ent_weight
                ] + gcn_mats + mini_batch['gcn_masks']
            )
        }

        return self._sess.run(
            [self._grads, self._all_loss, self._reward_loss, self._ent_loss, self._val_loss, self._is_ratio],
            feed_dict=feed_dict
        )

    def _aggregate_grads(self, all_grads):
        ground_grads = [np.zeros(grad.shape, dtype=np.float32) for grad in all_grads[0]]
        for grads in all_grads:
            for i, grad in enumerate(grads):
                ground_grads[i] += grad
        return ground_grads

    def apply_gradients(self, ground_grads, lr):
        feed_dict = {
            k: v for k, v in zip(
                self._grads + [self._lr],
                ground_grads + [lr]
            )
        }
        self._sess.run(self._train_op, feed_dict=feed_dict)

    def _build_get_vars_op(self):
        self._vars = tf.trainable_variables()

    def _build_set_vars_op(self):
        self._input_vars = []
        for _var in self._vars:
            self._input_vars.append(tf.placeholder(tf.float32, shape=_var.get_shape()))
        set_vars = []
        for i, _var in enumerate(self._vars):
            set_vars.append(_var.assign(self._input_vars[i]))
        self._set_vars_op = tf.group(set_vars)

    def get_vars(self):
        return self._sess.run(self._vars)

    def set_vars(self, input_vars):
        feed_dict = {
            k: v for k, v in zip(self._input_vars, input_vars)
        }
        self._sess.run(self._set_vars_op, feed_dict=feed_dict)

    def _expand_job_features(self, job_features):
        num_features = job_features.shape[2].value # deterministic
        expandding_features = tf.constant(
            np.linspace(0, 1, self._exec_cap, dtype=np.float32),
            name='exec_padding_features'
        )
        # (1, 1, exec_cap, 1)
        expandding_features = tf.reshape(expandding_features, [1, 1, -1, 1])
        # (batch_size, num_jobs, exec_cap, 1)
        expandding_features = tf.tile(expandding_features, [self._batch_size, self._num_jobs, 1, 1])

        # (batch_size, num_jobs, num_features * exec_cap)
        job_features = tf.tile(job_features, [1, 1, self._exec_cap])
        # (batch_size, num_jobs, exec_cap, num_features)
        job_features = tf.reshape(job_features, [self._batch_size, self._num_jobs, self._exec_cap, num_features])

        expanded_features = tf.concat([job_features, expandding_features], axis=-1)
        return expanded_features

    def save_model(self, global_step):
        saved_path = self._saver.save(self._sess, self._save_path, global_step=global_step)
        return saved_path
