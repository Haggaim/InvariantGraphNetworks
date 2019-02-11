from models.base_model import BaseModel
import layers.equivariant_linear as eq
import layers.layers as layers
import tensorflow as tf


class invariant_basic(BaseModel):
    def __init__(self, config, data):
        super(invariant_basic, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and define the loss.
        self.is_training = tf.placeholder(tf.bool)

        self.graphs = tf.placeholder(tf.float32, shape=[None, self.config.node_labels + 1, None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None])

        # build network architecture using config file
        net = eq.equi_2_to_2('equi0', self.data.train_graphs[0].shape[0], self.config.architecture[0], self.graphs)
        net = tf.nn.relu(net, name='relu0')
        for layer in range(1, len(self.config.architecture)):
            net = eq.equi_2_to_2('equi%d' %layer, self.config.architecture[layer-1], self.config.architecture[layer], net)
            net = tf.nn.relu(net, name='relu%d'%layer)

        net = layers.diag_offdiag_maxpool(net)

        net = layers.fully_connected(net, 512, "fully1")
        net = layers.fully_connected(net, 256, "fully2")
        net = layers.fully_connected(net, self.config.num_classes, "fully3", activation_fn=None)

        # define loss function
        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=net))
            self.correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(net, 1, output_type=tf.int32), self.labels), tf.int32))

        # get learning rate with decay every 20 epochs
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size*20)

        # choose optimizer
        if self.config.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_learning_rate(self, global_step, decay_step):
        """
        helper method to fit learning rat
        :param global_step: current index into dataset, int
        :param decay_step: decay step, float
        :return: output: N x S x m x m tensor
        """
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,  # Base learning rate.
            global_step*self.config.batch_size,
            decay_step,
            self.config.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate
