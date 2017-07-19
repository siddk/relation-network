"""
relation_network.py

Core model definition class for the Relation Network, by Santoro et. al. (Google Deepmind).
"""
import numpy as np
import tensorflow as tf


class RelationNetwork:
    def __init__(self, S, S_len, Q, Q_len, A, word2id, a_word2id, embed_size=32, lstm_size=32, batch_size=64,
                 initializer=tf.random_normal_initializer(stddev=0.1), ckpt_dir='ckpt/ckpts/', restore=False):
        """
        Initialize a Relation Network Model, with the necessary training data and hyperparameters.

        :param S: 3D Tensor object containing bAbI Stories [N, story_len, max_s]
        :param S_len: 2D Tensor object containing story line lengths [N, story_len]
        :param Q: 2D Tensor object containing queries [N, max_q]
        :param Q_len: 1D Tensor object containing query lengths [N]
        :param A: 1D Tensor object containing query answers [N]
        :param word2id: Vocabulary Dictionary mapping words to unique IDs (for stories, queries)
        :param a_word2id: Vocabulary mapping words to unique IDs (for answers)
        """
        self.S, self.S_len, self.Q, self.Q_len, self.A = S, S_len, Q, Q_len, A
        self.word2id, self.a_word2id = word2id, a_word2id
        self.story_len, self.max_s, self.max_q = self.S.shape[1], self.S.shape[2], self.Q.shape[1]
        self.embed_sz, self.lstm_sz, self.init, self.bsz = embed_size, lstm_size, initializer, batch_size
        self.ckpt_dir, self.restore = ckpt_dir, restore
        self.session = tf.Session()

        # Initialize Placeholders
        self.XS = tf.placeholder(tf.int64, shape=[None] + list(self.S.shape[1:]), name='Story')
        self.XS_len = tf.placeholder(tf.int64, shape=[None] + list(self.S_len.shape[1:]), name='Story_Length')
        self.XQ = tf.placeholder(tf.int64, shape=[None] + list(self.Q.shape[1:]), name='Query')
        self.XQ_len = tf.placeholder(tf.int64, shape=[None], name='Query_Length')
        self.YA = tf.placeholder(tf.int64, shape=[None], name='Answer')

        # Instantiate Weights
        self.instantiate_weights()

        # Build Inference Pipeline
        self.logits = self.inference()

        # Loss Computation
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.YA, self.logits)

        # Training Operation
        self.train_op = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.loss)

        # Create operations for computing the accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.YA)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        # Set up Epoch Step
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # Create Saver
        self.saver = tf.train.Saver()

        if self.restore:
            # Restore from Checkpoint
            self.session.run(tf.global_variables_initializer())
            self.saver.restore(self.session, self.restore)
        else:
            # Initialize all Variables
            self.session.run(tf.global_variables_initializer())

    def instantiate_weights(self):
        """
        Instantiate all Network Parameters, including Embedding Matrices for Story and Question Encoding.
        """
        # Create Story Embedding Matrix, with Zero Vector for PAD
        E = tf.get_variable("Embedding", [len(self.word2id), self.embed_sz], initializer=self.init)
        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(len(self.word2id))],
                                dtype=tf.float32, shape=[len(self.word2id), 1])
        self.E = E * zero_mask

        # Create Constant One-Hot Tags, to Denote Relative Position
        tag_constant = np.zeros([self.story_len, self.story_len], dtype=np.float32)
        for i in range(self.story_len):
            tag_constant[i][i] = 1
        self.tags = tag_constant

        # G_Theta Parameters
        self.G1_W = tf.get_variable("G1_W", shape=[(2*self.story_len) + (3*self.embed_sz), 256], initializer=self.init)
        self.G1_b = tf.get_variable("G1_b", shape=[256], initializer=self.init)

        self.G2_W = tf.get_variable("G2_W", shape=[256, 256], initializer=self.init)
        self.G2_b = tf.get_variable("G2_b", shape=[256], initializer=self.init)

        self.G3_W = tf.get_variable("G3_W", shape=[256, 256], initializer=self.init)
        self.G3_b = tf.get_variable("G3_b", shape=[256], initializer=self.init)

        self.G4_W = tf.get_variable("G4_W", shape=[256, 256], initializer=self.init)
        self.G4_b = tf.get_variable("G4_b", shape=[256], initializer=self.init)

        # F Parameters
        self.F1_W = tf.get_variable("F1_W", shape=[256, 256], initializer=self.init)
        self.F1_b = tf.get_variable("F1_b", shape=[256], initializer=self.init)

        self.F2_W = tf.get_variable("F2_W", shape=[256, 512], initializer=self.init)
        self.F2_b = tf.get_variable("F2_b", shape=[512], initializer=self.init)

        self.F3_W = tf.get_variable("F3_W", shape=[512, len(self.a_word2id)], initializer=self.init)
        self.F3_b = tf.get_variable("F3_b", shape=[len(self.a_word2id)], initializer=self.init)

    def inference(self):
        """
        Build inference pipeline, going from the story and question, through the relation module, to the
        distribution over possible answers.
        """
        # Encode Story Lines
        story_embeddings = tf.nn.embedding_lookup(self.E, self.XS)                  # Shape: [None, s_len, max_s, emb]
        flat_embeddings = tf.reshape(story_embeddings,                              # Shape: [None * s_len, max_s, emb]
                                     shape=[-1, self.max_s, self.embed_sz])
        flat_lengths = tf.reshape(self.XS_len, [-1])                                # Shape: [None * s_len]

        with tf.variable_scope("S_Encoder"):
            self.story_gru = tf.contrib.rnn.GRUCell(self.embed_sz)
            _, story_state = tf.nn.dynamic_rnn(self.story_gru, flat_embeddings,     # Shape: [None * s_len, emb]
                                               sequence_length=flat_lengths, dtype=tf.float32)
            s_obj = tf.reshape(story_state, shape=[-1, self.story_len,              # Shape: [None, s_len, emb]
                                                   self.embed_sz])
            s_tags = tf.tile(tf.expand_dims(self.tags, axis=0),
                             multiples=[tf.shape(s_obj)[0], 1, 1])                  # Shape: [None, s_len, 1]
            story_objects = tf.concat([s_tags, s_obj], axis=2)                      # Shape: [None, s_len, s_len + emb]

        # Encode Query
        query_embeddings = tf.nn.embedding_lookup(self.E, self.XQ)                  # Shape: [None, max_q, emb]
        with tf.variable_scope("Q_Encoder"):
            self.query_gru = tf.contrib.rnn.GRUCell(self.embed_sz)
            _, query_state = tf.nn.dynamic_rnn(self.query_gru, query_embeddings,    # Shape: [None, emb]
                                               sequence_length=self.XQ_len, dtype=tf.float32)
            query = tf.expand_dims(query_state, axis=1)                             # Shape: [None, 1, emb]

        # Do Pairwise Object-Query G_Theta Propagation
        sum_g_theta = 0
        objects = tf.split(story_objects, num_or_size_splits=self.story_len, axis=1)
        for i in range(self.story_len):
            for j in range(self.story_len):
                u = tf.squeeze(tf.concat([objects[i], objects[j], query], axis=2))  # Shape: [None, 2s_len + (3 * emb)]
                g = self.g_theta(u)
                sum_g_theta += g

        # Feed through F, Generate Logits
        h1 = tf.nn.relu(tf.matmul(sum_g_theta, self.F1_W) + self.F1_b)
        h2 = tf.nn.relu(tf.matmul(h1, self.F2_W) + self.F2_b)
        logits = tf.matmul(h2, self.F3_W) + self.F3_b

        return logits

    def g_theta(self, x):
        """
        MLP Helper Function for the G_Theta Object Logic.
        """
        h1 = tf.nn.relu(tf.matmul(x, self.G1_W) + self.G1_b)
        h2 = tf.nn.relu(tf.matmul(h1, self.G2_W) + self.G2_b)
        h3 = tf.nn.relu(tf.matmul(h2, self.G3_W) + self.G3_b)
        h4 = tf.nn.relu(tf.matmul(h3, self.G4_W) + self.G4_b)
        return h4

    def fit(self, epochs):
        """
        Train the model, with the specified batch size and number of epochs.
        """
        while self.session.run(self.epoch_step) < epochs:
            loss, acc, batches = 0.0, 0.0, 0
            for start, end in zip(range(0, len(self.S) - self.bsz, self.bsz), range(self.bsz, len(self.S), self.bsz)):
                c_loss, c_acc, _ = self.session.run([self.loss, self.accuracy, self.train_op],
                                                    feed_dict={self.XS: self.S[start:end],
                                                               self.XS_len: self.S_len[start:end],
                                                               self.XQ: self.Q[start:end],
                                                               self.XQ_len: self.Q_len[start:end],
                                                               self.YA: self.A[start:end]})
                loss, acc, batches = loss + c_loss, acc + c_acc, batches + 1
                if batches % 100 == 0:
                    print 'Epoch %d Batch %d\tAverage Loss: %.3f\tAverage Accuracy: %.3f' % (
                        self.session.run(self.epoch_step), batches, loss / batches, acc / batches)

            # Epoch Increment + Save
            self.session.run(self.epoch_increment)
            self.saver.save(self.session, self.ckpt_dir + "model.ckpt", global_step=self.epoch_step)

    def eval(self, evalS, evalS_len, evalQ, evalQ_len, evalA):
        """
        Evaluate the model on the given data.

        :param evalS: 3D Tensor object containing bAbI Stories [N, story_len, max_s]
        :param evalS_len: 2D Tensor object containing story line lengths [N, story_len]
        :param evalQ: 2D Tensor object containing queries [N, max_q]
        :param evalQ_len: 1D Tensor object containing query lengths [N]
        :param evalA: 1D Tensor object containing query answers [N]
        :return Accuracy (as float)
        """
        accuracy = self.session.run(self.accuracy, feed_dict={self.XS: evalS, self.XS_len: evalS_len, self.XQ: evalQ,
                                                              self.XQ_len: evalQ_len, self.YA: evalA})
        return accuracy

