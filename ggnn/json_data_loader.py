import os
import json
import gzip
import random
import math
import re

import threading
import multiprocessing
import time
from collections import Counter

random.seed(42)
import numpy as np

from graph import *


class GraphMetaData():
    def __init__(self, graph, properties):
        self.graph = graph
        self.properties = properties


class JSONDataLoader():
    edge_types = ["Child", "Parent", "PrevToken", "NextToken", "LastLexicalUse", "NextLexicalUse"]

    def __init__(self, data_config, train_root, test_root=None, vocab_path=None):
        self.max_batch_size = data_config["max_batch_size"]
        self.max_graph_size = data_config["max_graph_size"]
        self.split_tokens = data_config["split_tokens"]

        print("Loading train/test keys")
        self.train_keys = self.load_graph_paths(train_root, mode="training")
        random.shuffle(self.train_keys)
        if test_root != None:
            self.valid_keys = self.train_keys[int(0.95 * len(self.train_keys)):]
            self.train_keys = self.train_keys[:int(0.95 * len(self.train_keys))]
            self.test_keys = self.load_graph_paths(test_root, mode="testing")
        else:
            self.valid_keys = self.train_keys[int(0.95 * len(self.train_keys)):]
            self.test_keys = self.train_keys[int(0.9 * len(self.train_keys)):int(0.95 * len(self.train_keys))]
            self.train_keys = self.train_keys[:int(0.9 * len(self.train_keys))]
        self.get_vocabs(vocab_path, data_config["vocab_cutoff"])

    def get_vocabs(self, vocab_path, vocab_cutoff):
        if vocab_path != None:
            print("Reading vocabulary from base:", vocab_path)
            self.load_vocabs(vocab_path, vocab_cutoff)
        else:
            print("No vocabulary provided; building from training data")
            self.build_vocabs(vocab_cutoff)
        self.vocab_dim = len(self.w2i)
        self.vocab_key = lambda w: self.w2i[w] if w in self.w2i else self.w2i["<unk>"]  # Convenience function
        self.comment_vocab_key = lambda w: self.comment_w2i[w] if w in self.comment_w2i else self.comment_w2i["<unk>"]

    def load_vocabs(self, vocab_path, vocab_cutoff):
        with open(vocab_path, "r", encoding="utf8") as f:
            vocab = [l.rstrip('\n').split("\t") for l in f.readlines()]
            vocab = [l[1] for l in vocab if int(l[0]) >= vocab_cutoff]
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = {i: w for w, i in self.w2i.items()}
        if not "<unk>" in self.w2i:
            self.w2i["<unk>"] = len(self.w2i)
            self.i2w[self.w2i["<unk>"]] = "<unk>"

        with open(vocab_path + "-comment", "r", encoding="utf8") as f:
            vocab = [l.rstrip('\n').split("\t") for l in f.readlines()]
            vocab = [l[1] for l in vocab if int(l[0]) >= vocab_cutoff]
        self.comment_w2i = {w: i for i, w in enumerate(vocab)}
        self.comment_i2w = {i: w for w, i in self.comment_w2i.items()}
        if not "<unk>" in self.comment_w2i:
            self.comment_w2i["<unk>"] = len(self.comment_w2i)
            self.comment_i2w[self.comment_w2i["<unk>"]] = "<unk>"

    def tokenize_comment(self, comment):
        comment = re.findall(r"[\w']+|[.,!?;]", comment)
        words = []
        for comm in comment:
            words += util.split_subtokens(comm) if self.split_tokens else [
                comm.replace("\n", "").replace("\r", "")]
        return words

    def build_vocabs(self, vocab_cutoff, write=True):
        token_counts = {}
        comment_token_counts = {}
        self.w2i = None
        self.comments_w2i = None
        for key in self.train_keys:
            print("Reading", key)
            for graph in self.load_graph_data(key):
                node_labels = graph["Nodes"]
                node_comments = graph["Comments"]

                # Normal token splitting
                if len(node_labels) > self.max_graph_size: continue
                for label in node_labels.values():
                    subtokens = util.split_subtokens(label) if self.split_tokens else [
                        label.replace("\n", "").replace("\r", "")]
                    for sub in subtokens:
                        if sub in token_counts:
                            token_counts[sub] += 1
                        else:
                            token_counts[sub] = 1

                # Comment Token Spitting
                for comment in node_comments.values():
                    if comment == "":
                        continue
                    words = self.tokenize_comment(comment)
                    for word in words:
                        if word in comment_token_counts:
                            comment_token_counts[word] += 1
                        else:
                            comment_token_counts[word] = 1

        # Ensure some key tokens make it into the vocabulary
        if "<unk>" not in token_counts: token_counts["<unk>"] = max(vocab_cutoff, sum(
            [c for c in token_counts.values() if c < vocab_cutoff]))

        # Sort and discard tokens to infrequent to keep
        top_words = sorted(token_counts.items(), key=lambda t: t[1], reverse=True)
        top_words = [t[0] for t in top_words if t[1] >= vocab_cutoff]

        # Build the vocabulary
        self.w2i = {w: i for i, w in enumerate(top_words)}
        self.i2w = {i: w for w, i in self.w2i.items()}

        # # Ensure some key tokens make it into the comment vocabulary
        if "<unk>" not in comment_token_counts: comment_token_counts["<unk>"] = max(vocab_cutoff, sum(
            [c for c in comment_token_counts.values() if c < vocab_cutoff]))

        # Sort and discard comment tokens to infrequent to keep
        top_comment_words = sorted(comment_token_counts.items(), key=lambda t: t[1], reverse=True)
        top_comment_words = [t[0] for t in top_comment_words if t[1] >= vocab_cutoff]

        # Build the comment vocabulary
        self.comment_w2i = {w: i for i, w in enumerate(top_comment_words)}
        self.comment_i2w = {i: w for w, i in self.comment_w2i.items()}

        if not write: return
        with open("vocab", "w", encoding="utf8") as f:
            for ix in range(len(self.w2i)):
                w = self.i2w[ix]
                f.write(str(token_counts[w]))
                f.write("\t")
                f.write(w)
                if ix < len(self.w2i) - 1: f.write('\n')

        if not write: return
        with open("vocab-comment", "w", encoding="utf8") as f2:
            for ix in range(len(self.comment_w2i)):
                w = self.comment_i2w[ix]
                f2.write(str(comment_token_counts[w]))
                f2.write("\t")
                f2.write(w)
                if ix < len(self.comment_w2i) - 1: f2.write('\n')

    def batcher(self, mode="training"):

        # Adds sub-token indices (if applicable) and their corresponding node location to the graph
        def add_indices(graph):
            indices = []
            locs = []
            loc = 0
            for node in graph.nodes.values():
                tokens = node.subtokens if self.split_tokens else [node.label]
                indices.extend([self.vocab_key(t) for t in tokens])
                locs.extend(len(tokens) * [loc])
                loc += 1
            graph.indices = indices
            graph.locs = locs
            return graph

        # tokenizes the comments and creates two square matrix, indexes and maskes
        def create_comment_tensor(comments):
            just_comments = [list(comment.values())[0] for comment in comments]
            keys = [int(list(comment.keys())[0]) for comment in comments]
            comments = [self.tokenize_comment(comm) for comm in just_comments]

            indices = []
            masks = []
            max_len = max([len(comment) for comment in comments])
            for cix, comment in enumerate(comments):
                masks.append([0] * len(comment) + [1] * (max_len - len(comment)))
                indices.append([self.comment_vocab_key(comm) for comm in comment] + [1] * (max_len - len(comment)))
            return np.array(indices), np.array(masks), keys

        # Samples a random subset of similarly sized items from the current queue of graphs
        def sample_batch(queue):
            random.shuffle(queue)
            sample_ixs = set()
            batch_size = 0
            for ix, g in enumerate(queue):
                if batch_size + g[0].size() > self.max_batch_size: break
                sample_ixs.add(ix)
                batch_size += g[0].size()
            sample = [queue[ix] for ix in sample_ixs]
            rest = [queue[ix] for ix in range(len(queue)) if ix not in sample_ixs]
            graphs = [add_indices(graph) for graph, _ in sample]
            comment_indexes, comment_mask, comment_keys = create_comment_tensor([comm for _, comm in sample])
            return rest, (graphs, comment_indexes, comment_mask, comment_keys)

        keys = self.train_keys if mode == "training" else self.test_keys if mode == "testing" else self.valid_keys
        if mode == "training": random.shuffle(keys)
        graph_queue = []
        for key in keys:
            samples = self.load_graph_file(key)
            if mode == "training": random.shuffle(samples)
            for sample in samples:
                graph = sample.graph
                comment = sample.properties["Comments"]
                if sum([g[0].size() for g in graph_queue]) > 10 * self.max_batch_size:
                    graph_queue, sample = sample_batch(graph_queue)
                    yield sample
                graph_queue.append((graph, comment))

        while len(graph_queue) > 0:
            graph_queue, sample = sample_batch(graph_queue)
            yield sample

    """ Various functions for loading graph data """

    def load_graph_paths(self, path, mode="training"):
        if os.path.isfile(path) and (path.endswith(".json") or path.endswith(".gz")):
            return [path]
        else:
            keys = []
            for child in os.listdir(path):
                f = os.path.join(path, child)
                keys.extend(self.load_graph_paths(f))
            return keys

    def load_graph_data(self, graph_file):
        if graph_file.endswith(".gz"):
            with gzip.open(graph_file, "rb") as f:
                raw_data = json.loads(f.read().decode())
        else:
            with open(graph_file, "r", encoding="utf-8") as f:
                raw_data = json.loads(f.read())
        return raw_data

    def load_graph_file(self, graph_file, mode="training"):
        raw_data = self.load_graph_data(graph_file)

        graph_datas = []
        for json in raw_data:
            data = self.load_graph(json, mode)
            if data == None:
                continue
            else:
                graph_datas.append(data)
        return graph_datas

    def load_graph(self, json, mode="training"):
        # Only train on graphs that have comments
        if len(json["Comments"]) == 0: return None

        # Load nodes
        node_labels = json["Nodes"]
        if len(node_labels) > self.max_graph_size: return None
        nodes = {}
        for k, v in node_labels.items():
            node = Node(int(k), v)
            nodes[int(k)] = node

        # Load edges
        edge_categories = json["Edges"]
        edges = list()  # Just store as list; Graph class will index these
        for edge_type in edge_categories.keys():
            for edge in edge_categories[edge_type]:
                if edge[0] == {}: print("Erroneous edge?", json)
                if edge_type not in self.edge_types: continue
                if edge[0] not in nodes or edge[1] not in nodes: continue
                ix = self.edge_types.index(edge_type)
                reverse_type = self.edge_types[ix + 1 if ix % 2 == 0 else ix - 1]
                edges.append(Edge(edge_type, edge[0] - 1, edge[1] - 1))
                edges.append(Edge(reverse_type, edge[1] - 1, edge[0] - 1))

        graph = Graph(nodes, edges)
        return GraphMetaData(graph, {k: json[k] for k in json.keys() if
                                     k not in ["Nodes", "Edges"]})  # Store all meta-data (e.g. method-name, comments)
