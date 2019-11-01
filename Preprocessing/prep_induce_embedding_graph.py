import numpy as np
import tensorflow as tf
import json
import networkx as nx
from networkx.readwrite import json_graph as jg
import prep_preformatted as pp
from matplotlib import pyplot as plt
import os
from prep_data_gs import dumpJSON
from utils import load_data as load
import induce_graph

try:
    import queue
except ImportError:
    import Queue as queue


flags = tf.app.flags
FLAGS = flags.FLAGS

'''
# Dataset and destination directory
flags.DEFINE_string('train_prefix', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_100/cora', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('embedding_dir', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_100/cora_unsupervised/unsup-embedding_pre/graphsage_mean_small_0.000010', 'Directory to which the entire training embeddings are stored.')
flags.DEFINE_integer('random_seed_num', 30, 'in this experiment, we use num_train/10 by default')
'''

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"


def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    # print G.nodes()[0]
    # check G.nodes()[0] is an integer or not
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n
    # Number of orgin samples
    num_origin_nodes = G.number_of_nodes()
    print("The number of original nodes is %d"%(num_origin_nodes))

    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    # just print the id_map keys range:
    # id_map_range = np.sort(id_map.keys())
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}




    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    return G, id_map, class_map, feats

def induce_embedding(train_idx, G, id_map, class_map, feats,data_dir,val_size,test_size):
    # Number of training samples to be kept
    num_train = int(train_idx * FLAGS.train_percent)
    num_nodes = len(G.node)


    # load the embedding of processed all training instances
    #/val.txt means the id of the node
    #/val.npy is the embedding for the corresponding id for /val.txt
    embeds = np.load(data_dir + "/val.npy")
    id_map = {}
    with open(data_dir + "/val.txt") as fp:
        for i, line in enumerate(fp):
            id_map[int(line.strip())] = i
    #adjust the correct order of mapping of train_embeds starting from node 0
    train_embeds = embeds[[id_map[id] for id in range(train_idx)]]

    # Initialize queue and list of nodes that have been visited
    Q = queue.Queue()
    seenNodes = []
    firstNeigh_train_node = queue.Queue()
    firstNeigh_train_list =[]

    # Add testing and val Nodes to Queue
    for n in range(num_nodes - (test_size + val_size)-1, num_nodes):
        Q.put(list(G)[n])
        seenNodes.append(list(G)[n])

    # The number of testing nodes currently in the queue is equal to the total
    # The number of first order neighbors in the queue is zero
    testNum = test_size + val_size
    firstNeigh = 0

    # Keep going until there are neither testing nodes nor first neighbors in the queue
    while testNum > 0:

        # Pop the first node
        node = Q.get()

        # If it was a testing node, decrement testNum
        if testNum > 0:
            testNum = testNum - 1

        # Iterate through each neighbor of current node (BFS)
        for neighbor in G.neighbors(node):
            if G.node[neighbor]['test']==False and G.node[neighbor]['val']==False:
                if neighbor not in firstNeigh_train_list:
                    firstNeigh_train_node.put(neighbor)
                    firstNeigh_train_list.append(neighbor)
    print("The size of the firstNeigh_train is %d" %(firstNeigh_train_node.qsize()))

    # Arrays of the indices of polluted nodes in each of the three sets
    keep_idx_train = np.random.choice(firstNeigh_train_list, FLAGS.random_seed_num, replace=False)








def main():
    valNum = 500
    testNum = 1000

    # Load data
    G, IDMap, classMap, features = load_data(FLAGS.train_prefix)

    # Index of the highest training node
    trainIdx = G.number_of_nodes() - (valNum + testNum) - 1

    # Generate the induced graph
    G, IDMap, classMap, features = induce_embedding(trainIdx, G, IDMap, classMap, features,FLAGS.embedding_dir,valNum,testNum)



if __name__ == "__main__":
    main()

