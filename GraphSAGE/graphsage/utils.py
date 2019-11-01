from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

import sys
sys.path.insert(1, '/Users/april/Downloads/GraphSAGE_Benchmark-master/Preprocessing')


#import prep_data_embedding as pde

def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    print("The number of edges")
    edge_num = G.number_of_edges()
    print(edge_num)
    print("The number of nodes")
    nodes_num = G.number_of_nodes()
    print(nodes_num)
    #print G.nodes()[0]
    #check G.nodes()[0] is an integer or not
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    #just print the id_map keys range:
    #id_map_range = np.sort(id_map.keys())
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}


    # just print the class_map keys range:
    class_map_int_list =[]
    for j in class_map.keys():
        class_map_int_list.append(int(j))
    class_map_range = np.sort(class_map_int_list)

    #print the anormaly ground truth number
    anormaly_count_gt = 0
    anormaly_count_vl = 0
    anormaly_count_tn = 0
    for node in G.nodes():
        if G.node[node]['test']==True:
            if G.node[node]['label']==[0,1]:
                anormaly_count_gt+=1
        if G.node[node]['val']==True:
            if G.node[node]['label']==[0,1]:
                anormaly_count_vl+=1
        if G.node[node]['val']!=True and G.node[node]['test']!=True:
            if G.node[node]['label']==[0,1]:
                anormaly_count_tn+=1
    print("anormaly in test data is %d"%(anormaly_count_gt))
    print("anormaly in validation data is %d" % (anormaly_count_vl))
    print("anormaly in training data is %d" %(anormaly_count_tn))

    node_degrees = list(G.degree().values())
    print ("the maximum degree of the graph is %d" %max(node_degrees))



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
    #add the train_removed Flag for each edge in G.edges
    #temp_useful_edges =0
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
            #temp_useful_edges+=1
    #print (G.node[edge[0]])
    #print ("The real edges that are taken account in is %d" %(temp_useful_edges))
    #1432 useful edges marked with train_removed = False

    ''' Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean 
    and standard deviation are then stored to be used on later data using the transform method. 
    If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn 
    from other features correctly as expected.
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

    return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    #graph_file = sys.argv[1]
    #out_file = sys.argv[2]
    #print (sys.argv)

    #####Only works when needs generating walks
    graph_file =pde.FLAGS.embedding_destination_dir+"/cora"+"-G.json"
    out_file = pde.FLAGS.embedding_destination_dir+"/cora"+"-walks.txt"
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
