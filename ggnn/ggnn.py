import os
import sys
import json
import gzip
import time
import random

import numpy as np
import tensorflow as tf

from graph import *
import util

class GraphModel(tf.keras.layers.Layer):
	def __init__(self, config, vocabulary, edge_types):
		super(GraphModel, self).__init__()
		self.vocabulary = vocabulary
		self.edge_types = edge_types
		
		self.time_steps = config["ggnn"]["time_steps"]
		self.num_layers = len(self.time_steps)
		self.residuals = config["ggnn"]["residuals"]
		self.hidden_dim = config["ggnn"]["hidden_dim"]
		self.make_layers()
	
	def make_layers(self):
		# Small util functions
		random_init = tf.random_normal_initializer(stddev=0.1)
		def make_weight(name=None):
			return tf.Variable(random_init([self.hidden_dim, self.hidden_dim]), name=name)
		def make_bias(name=None):
			return tf.Variable(random_init([self.hidden_dim]), name=name)
		
		# Set up embedding, type-transform and rnn layers
		self.embedding = tf.Variable(random_init([len(self.vocabulary), self.hidden_dim]), dtype=tf.float32)
		self.type_weights = [[make_weight("type-" + str(j) + "-" + str(i)) for i in range(len(self.edge_types))] for j in range(self.num_layers)]
		self.type_biases = [[make_bias("bias-" + str(j) + "-" + str(i)) for i in range(len(self.edge_types))] for j in range(self.num_layers)]
		self.rnns = [tf.keras.layers.GRUCell(self.hidden_dim) for _ in range(self.num_layers)]
		for ix, rnn in enumerate(self.rnns):
			if ix == len(self.time_steps): rnn.build(2*self.hidden_dim)
			elif ix not in self.residuals: rnn.build(self.hidden_dim)
			else: rnn.build(self.hidden_dim*(1 + len(self.residuals[ix])))
	
	def call(self, graphs):
		# Initialize the node_states with subtoken (and possibly type) embeddings
		layer_states = [self.embed_nodes(graphs)]
		
		# Propagate through the layers and the number of time steps for each layer
		for layer_no, steps in enumerate(self.time_steps):
			for step in range(steps):
				residual = None if layer_no not in self.residuals else [layer_states[ix] for ix in self.residuals[layer_no]]
				new_states = self.propagate(layer_no, graphs, layer_states[-1], residual=residual)
				if step == 0: layer_states.append(new_states)
				else: layer_states[-1] = new_states
		
		return layer_states[-1]
	
	# Embed all indices in parallel, then aggregate by node locations (takes care of sub-tokenized embeddings)
	def embed_nodes(self, graphs):
		sum_sizes = util.prefix_sum([graph.size() for graph in graphs])
		node_locs = [sum_sizes[gix] + loc for gix, graph in enumerate(graphs) for loc in graph.locs]
		node_ids = tf.concat([graph.indices for graph in graphs], axis=0)
		embs = tf.nn.embedding_lookup(self.embedding, node_ids)
		embs = tf.math.segment_sum(data=embs, segment_ids=node_locs)
		return embs
	
	def propagate(self, layer_no, graphs, states, residual=None):
		# Collect some basic details about the graphs in the batch, for shared learning (don't record this)
		sum_sizes = util.prefix_sum([graph.size() for graph in graphs])
		
		# Collect messages for all edge types
		messages = []
		message_targets = []
		for type in self.edge_types:
			# Collect edge sources and targets for this edge type (don't record)
			edge_sources = []
			for gix, graph in enumerate(graphs):
				if type not in graph.type_edges: continue
				for edge in graph.type_edges[type]:
					edge_sources.append(sum_sizes[gix] + edge.source_ix)
					message_targets.append(sum_sizes[gix] + edge.target_ix)
			if len(edge_sources) == 0: continue
			
			# If any, retrieve source states and compute transformations
			type_ix = self.edge_types.index(type)
			edge_source_states = tf.nn.embedding_lookup(states, edge_sources)
			type_messages = tf.matmul(edge_source_states, self.type_weights[layer_no][type_ix]) + self.type_biases[layer_no][type_ix]
			messages.append(type_messages)
		
		# Group and sum messages by target.
		messages = tf.concat(messages, axis=0)
		messages = tf.math.unsorted_segment_sum(data=messages, segment_ids=message_targets, num_segments=sum_sizes[-1])
		
		# Get residual messages, if applicable
		if not residual is None: messages = tf.concat(residual + [messages], axis=1)
		
		# Run RNN for each node
		new_states, _ = self.rnns[layer_no](messages, tf.expand_dims(states, 0))
		return new_states
