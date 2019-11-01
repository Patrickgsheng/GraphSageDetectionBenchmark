import numpy as np
from utils import load_data as load
import tensorflow as tf
import json
import networkx as nx
from networkx.readwrite import json_graph as jg
import induce_graph as ig
import prep_preformatted as pp
from matplotlib import pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

# Dataset and destination directory
flags.DEFINE_string('dataset', 'cora', 'Dataset to be used (citeseer/cora).')
flags.DEFINE_string('destination_dir', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_100/', 'Directory to which the data files will be sent.')
flags.DEFINE_string('induce_method', 'BFS', 'By what method to induce the subgraph: BFS, rand, or None')

##
# Returns a graph, constructed out of the inputted adjacency, label, and feature matrices, 
# as well as the test and validation masks.
##
def create_G_idM_classM(adjacency, features, testMask, valMask, labels):
    
    # 1. Create Graph
    print("Creating graph...")
    # Create graph from adjacency matrix
    # G = nx.from_numpy_array(adjacency)
    #To make the package dependency consistent with NetworkX 1.11
    G = nx.from_numpy_matrix(adjacency)
    num_nodes = G.number_of_nodes()
    
    # Change labels to int from numpy.int64
    labels = labels.tolist()
    for arr in labels:
        for integer in arr:
            integer = int(integer)
    
    # Iterate through each node, adding the features
    i = 0
    for n in list(G):
        G.node[i]['feature'] = list(map(float, list(features[i])))
        G.node[i]['test'] = bool(testMask[i])
        G.node[i]['val'] = bool(valMask[i])
        G.node[i]['labels'] = list(map(int, list(labels[i])))
        i += 1
       
    # 2. Create id-Map and class-Map
    print("Creating id-Map and class-Map...")
    # Initialize the dictionarys
    idM = {}
    classM = {}
    
    # Populate the dictionarys
    i = 0
    while i < num_nodes:
        idStr = str(i)
        idM[idStr] = i
        classM[idStr] = list(labels[i])
        i += 1
    
    return G, idM, classM
    

##
# Dumps the inputted graph, id-map, class-map, and feature matrix into their respective
# .json files and .numpy file.
##
def dumpJSON(destDirect, datasetName, graph, idMap, classMap, features):
    
    print("Dumping into JSON files...")
    # Turn graph into data
    dataG = jg.node_link_data(graph)
    
    #Make names
    json_G_name = destDirect + '/' + datasetName + '-G.json'
    json_ID_name = destDirect + '/' + datasetName + '-id_map.json'
    json_C_name = destDirect + '/' + datasetName + '-class_map.json'
    npy_F_name = destDirect + '/' + datasetName + '-feats'
    
    # Dump graph into json file
    with open(json_G_name, 'w') as outputFile:
        json.dump(dataG, outputFile)
        
    # Dump idMap into json file
    with open(json_ID_name, 'w') as outputFile:
        json.dump(idMap, outputFile)
        
    # Dump classMap into json file
    with open(json_C_name, 'w') as outputFile:
        json.dump(classMap, outputFile)
        
    # Save features as .npy file
    print("Saving features as numpy file...")
    np.save(npy_F_name, features)


def main():


    valNum = 500
    testNum = 1000

    # Load data
    adj, features, labels, valMask, testMask = load(FLAGS.dataset)
    
    # Turn CSR matricies into numpy arrays
    adj = adj.toarray()
    features = features.toarray()
    
    # Create Graph, IDMap, and classMap
    G, IDMap, classMap = create_G_idM_classM(adj, features, testMask, valMask, labels)

    print("\nORIGINAL GRAPH~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(G.number_of_nodes())
    #degrees = [val for (node, val) in G.degree()]
    degrees = list(G.degree().values())
    # print("Number of isolated nodes:", degrees.count(0))
    '''options = {
            'arrows': True,
            'node_color': 'blue',
            'node_size': .05,
            'line_color': 'black',
            'linewidths': 1,
            'width': 0.1,
            'with_labels': False,
            'node_shape': '.',
            'node_list': range((G.number_of_nodes() - 1500), G.number_of_nodes())
            }
    coef = nx.average_clustering(G)
    print("Average clustering coefficient:", coef)
    print("Density: ", nx.density(G))
    #nx.draw_networkx(G, **options)
    #plt.savefig('firstGraph.png', dpi=1024)'''

    # Index of the highest training node
    trainIdx = G.number_of_nodes() - (valNum + testNum) - 1


    # Induce Graph
    if FLAGS.induce_method == 'rand':
        G, IDMap, classMap, features = ig.induce_rand(trainIdx, G, IDMap, classMap, features)

    elif FLAGS.induce_method == 'BFS':
        G, IDMap, classMap, features = ig.induce_BFS(trainIdx, G, IDMap, classMap, features, valNum, testNum)



    print("\nNEW GRAPH~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(G.number_of_nodes())
    options = {
        'arrows': True,
        'node_color': 'blue',
        'node_size': .05,
        'line_color': 'black',
        'linewidths': 1,
        'width': 0.1,
        'with_labels': False,
        'node_shape': '.'
    }
    #degrees = [val for (node, val) in G.degree()]
    degrees = list(G.degree().values())
    print("Number of isolated nodes:", degrees.count(0))
    #Save all the degrees =0 in the new processed graph for analysis
    chosen_id = [i for i in range(len(degrees)) if degrees[i]==0]
    np.savetxt(FLAGS.destination_dir + '/' +'degree_equal_0_idlist',
               chosen_id, delimiter=",")
    coef = nx.average_clustering(G)
    print("Average clustering coefficient:", coef)
    print("Density: ", nx.density(G))
    #nx.draw_networkx(G, **options)
    #plt.savefig(FLAGS.destination_dir + '/vis.png', dpi=1024)

    # Pollute the graph
    trainIdx, G, classMap, features = pp.pollute_graph(G, IDMap, classMap, features, valNum, testNum)
    
    # Dump everything into .json files and one .npy
    dumpJSON(FLAGS.destination_dir, FLAGS.dataset, G, IDMap, classMap, features)
    
    
if __name__ == "__main__":
    main()
