import sys
import os
import math
import random
import yaml
import argparse
import time

import numpy as np
import tensorflow as tf

from json_data_loader import JSONDataLoader
from ggnn import GraphModel
# from transformer import TransformerModel, AttentionLayer
# from metrics import MetricsTracker
import util

random.seed(41)
config = yaml.safe_load(open("config.yml"))


def main():
    # Extract arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("train_data", help="Path to training data")
    ap.add_argument("-t", "--test_data", required=False, help="Path to test data")
    ap.add_argument("-v", "--vocab", required=False, help="Path to vocabulary file")
    args = ap.parse_args()

    print("Using configuration:", config)
    data = JSONDataLoader(config["data"], args.train_data, test_root=args.test_data, vocab_path=args.vocab)
    model = CommentPredictionModel(GraphModel(config, data.w2i, data.edge_types), config["comment"],
                                   len(data.comment_w2i))
    train(model, data)


def train(model, data):
    # Declare the learning rate as a variable to include it in the saved state
    learning_rate = tf.Variable(config["training"]["lr"], name="learning_rate")
    optimizer = tf.optimizers.Adam(learning_rate)

    is_first = True
    for epoch in range(config["training"]["num_epochs"]):
        print("Epoch:", epoch + 1)
        mbs = num_graphs = num_samples = 0
        avg_loss = avg_acc = avg_count = 0
        for graph_tuples in data.batcher(mode="training"):
            mbs += 1
            num_graphs += len(graph_tuples[0])
            num_samples += sum([graph.size() for graph in graph_tuples[0]])
            # Run through one batch to init variables
            if is_first:
                model(*graph_tuples)
                is_first = False
                print("Model initialized, training {:,} parameters".format(
                    np.sum([np.prod(v.shape) for v in model.trainable_variables])))

            # Compute loss in scope of gradient-tape (can also use implicit gradients)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_variables)
                choice, loss, acc = model(*graph_tuples)
                loss = tf.reduce_sum(loss)

            # Collect gradients, clip and apply
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.25)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            avg_loss += loss
            avg_acc += acc
            avg_count += len(graph_tuples[0])
            if mbs % config["training"]["print_freq"] == 0:
                print("At MB {0}, {1} graphs, {2} nodes: loss={3:.3f}, accuracy={4:.2%}".format(mbs, num_graphs,
                                                                                                num_samples,
                                                                                                avg_loss / avg_count,
                                                                                                avg_acc / avg_count))
                avg_loss = avg_acc = avg_count = 0

        valid_samples, valid_loss, valid_accuracy = eval(model, data)
        print("Validation: {0} comments, loss={1:.3f}, accuracy={2:.2%}".format(valid_samples, valid_loss,
                                                                                valid_accuracy))
        test_samples, test_loss, test_accuracy = eval(model, data, validate=False)
        print("Test: {0} comments, loss={1:.3f}, accuracy={2:.2%}".format(test_samples, test_loss, test_accuracy))


def eval(model, data, validate=True):
    total_loss = 0.0
    total_accuracy = 0.0
    comment_count = 0
    for graph_tuples in data.batcher(mode="validating" if validate else "testing"):
        choice, loss, acc = model(*graph_tuples)
        total_loss += tf.reduce_sum(loss)
        total_accuracy += acc
        comment_count += len(graph_tuples[0])
    return comment_count, total_loss / comment_count, total_accuracy / comment_count


class CommentPredictionModel(tf.keras.layers.Layer):
    def __init__(self, ggnn, config, comment_vocab_dim):
        super(CommentPredictionModel, self).__init__()
        self.ggnn = ggnn
        self.comment_dim = config["hidden_dim"]
        self.scale_up = tf.keras.layers.Dense(
            self.comment_dim)  # GGNN's default hidden state is probably too low for comment encoder; scaling it up to e.g. 512 should make it easier
        self.embed = tf.Variable(tf.random_normal_initializer(stddev=0.1)([comment_vocab_dim, self.comment_dim]),
                                 dtype=tf.float32)
        self.rnn = tf.keras.layers.GRU(self.comment_dim, return_sequences=True)

    def call(self, graphs, comment_indexes, comment_masks, comment_keys):
        # Run GGNNs on graphs. Returns hidden states for all nodes in all graphs in one "flat" tensor
        states = self.ggnn(graphs)
        states = self.scale_up(states)

        # Extract the comment text and location values to compare with the GGNN's states
        sum_sizes = util.prefix_sum([graph.size() for graph in graphs])

        comment_states = tf.nn.embedding_lookup(self.embed, comment_indexes)
        comment_states = self.rnn(comment_states)
        comment_states *= tf.cast(tf.expand_dims(comment_masks, -1), "float32")
        comment_states = tf.reduce_sum(comment_states, axis=1)
        comment_states *= tf.math.rsqrt(tf.cast(self.comment_dim, "float32"))
        comment_states = tf.concat(
            [tf.tile(x, [graphs[ix].size(), 1]) for ix, x in enumerate(tf.split(comment_states, len(graphs)))], axis=0)

        comment_indices = [[sum_size + comment_keys[gix]] for gix, sum_size in enumerate(sum_sizes[:-1])]
        choice = tf.reduce_sum(states * comment_states, -1)
        choice = tf.concat(
            [tf.nn.softmax(x) for ix, x in enumerate(tf.split(choice, [graph.size() for graph in graphs]))], axis=0)
        comment_key_tensor = tf.scatter_nd(indices=comment_indices, updates=[1.0] * len(graphs), shape=choice.shape)
        loss = comment_key_tensor * -tf.math.log(1e-6 + choice)

        accuracy = sum([1 if tf.argmax(x).numpy() == comment_keys[ix] else 0 for ix, x in
                        enumerate(tf.split(choice, [graph.size() for graph in graphs]))])
        return choice, tf.gather_nd(loss, comment_indices), accuracy


if __name__ == '__main__':
    main()
