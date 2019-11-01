import numpy as np
import tensorflow as tf
import json
import networkx as nx
from networkx.readwrite import json_graph as jg
import induce_graph as ig
import prep_preformatted as pp
from matplotlib import pyplot as plt
import os
from prep_data_gs import dumpJSON


import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

flags = tf.app.flags
FLAGS = flags.FLAGS

# Dataset and destination directory
flags.DEFINE_string('embedding_dataset', 'cora', 'Dataset to be used (citeseer/cora).')
flags.DEFINE_string('embedding_destination_dir', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_100/embedding_pre', 'Directory to which the data files will be sent.')
flags.DEFINE_string('train_prefix', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_100/cora', 'prefix identifying training data. must be specified.')



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


    for node in G.nodes():
        if G.node[node]['test'] == False and G.node[node]['val']==False:
            print("keep the node %d for embedding learning" %(node))
        else:
            print("Remove the node %d since validation/testing" %(node))
            G.remove_node(node)
            id_map.pop(node)
            class_map.pop(node)


    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    featsN = np.array([feats[0]])
    for n in list(G):
        row = G.node[n]['feature']
        featsN = np.append(featsN, [row], axis=0)
    featsN = np.delete(featsN, 0, 0)



    '''
    # just print the class_map keys range:
    class_map_int_list =[]
    for j in class_map.keys():
        class_map_int_list.append(int(j))
    class_map_range = np.sort(class_map_int_list)

    #print the anormaly ground truth number
    anormaly_count_gt = 0
    for node in G.nodes():
        if G.node[node]['test']==True:
            if G.node[node]['label']==[0,1]:
                anormaly_count_gt+=1
    print("anormaly in test data is %d"%(anormaly_count_gt))
    

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    # add the train_removed Flag for each edge in G.edges
    temp_useful_edges =0
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
            temp_useful_edges+=1
    # print (G.node[edge[0]])
    # print ("The real edges that are taken account in is %d" %(temp_useful_edges))
    # 1432 useful edges marked with train_removed = False
    temp_edge_num = 0
    for edge in G.edges():
        for i in edge:
            temp_edge_num+=1
    print("The # of edges in G is %d"%(temp_edge_num))
    '''

    ''' Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean 
    and standard deviation are then stored to be used on later data using the transform method. 
    If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn 
    from other features correctly as expected.
    '''
    '''
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))
    '''

    return G, id_map, class_map, featsN




def main():

    # Load data
    G,id_map,class_map,featsN = load_data(FLAGS.train_prefix)
    # dump the preprocessed G, id_map, class_map, featsN
    # Dump everything into .json files and one .npy
    dumpJSON(FLAGS.embedding_destination_dir, FLAGS.embedding_dataset, G, id_map, class_map, featsN)




if __name__ == "__main__":
    main()