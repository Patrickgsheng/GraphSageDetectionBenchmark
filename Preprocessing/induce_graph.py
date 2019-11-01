import numpy as np
import tensorflow as tf
import networkx as nx
try:
    import queue
except ImportError:
    import Queue as queue

flags = tf.app.flags
FLAGS = flags.FLAGS

# 
flags.DEFINE_float('train_percent', 1.0, 'Percentage of training data to be used.')
    

##
# Trim training graph down to a speficied percentage of it's original size, completely randomly
##
def trim_graph(train_idx, keep_idx_train, G, id_map, class_map, feats):
    
    graphSizeOriginal = G.number_of_nodes()
    keep_idx_train.sort()
    print("The number of kept training nodes is %d" %(len(keep_idx_train)))
    
    # Iterate through each node index in the training graph
    for i in train_idx:
        # if the node is not in the "keep" list, remove it from the three objects
        if i not in keep_idx_train:
            
            G.remove_node(i)
            id_map.pop(str(i))
            class_map.pop(str(i)) 
        
    featsN = np.array([feats[keep_idx_train[0]]])    
    for n in list(G):
        print ("currrent n is %d" %n)
        row = G.node[n]['feature']
        featsN = np.append(featsN, [row], axis=0)
    featsN = np.delete(featsN, 0, 0)
                 
    # Reindex the nodes
    id_map = {}
    new_class_map = {}
    new_label = {}


    i = 0
    for n in G: 
        new_label[n] = i
        id_map[str(i)] = i
        i += 1

    G = nx.relabel_nodes(G, new_label)

    #Regenerate the class_map after the node reindex
    i = 0
    for n in class_map:
        new_class_map[str(i)] = class_map[n]
        i += 1


    # save the id mapping information, from the original graph, what nodes are kept is clear
    np.savetxt( FLAGS.destination_dir + '/' +FLAGS.datasetname+'label_map',
             keep_idx_train, delimiter=",")

    print("trim process is finished!!!!")
    return G, id_map, new_class_map, featsN

##
# Return indicies to be retained from the train set, using BFS from test set
##
def induce_BFS(train_idx, G, id_map, class_map, feats, val_size, test_size):
    
    # Number of training samples to be kept
    num_nodes = G.number_of_nodes()
    num_train_keep = int(train_idx * FLAGS.train_percent)
    
    
    # Initialize queue and list of nodes that have been visited
    Q = queue.Queue()
    seenNodes = []
    
    # Add testing and val Nodes to Queue
    for n in range(num_nodes - (test_size+val_size), num_nodes):
        Q.put(list(G)[n])
        seenNodes.append(list(G)[n])
        
    # The number of testing nodes currently in the queue is equal to the total
    # The number of first order neighbors in the queue is zero
    testNum = test_size + val_size
    firstNeigh = 0
    
    # Keep going until there are neither testing nodes nor first neighbors in the queue
    while firstNeigh > 0 or testNum > 0:
        
        # Pop the first node
        node = Q.get()
        
        # If it was a testing node, decrement testNum
        if testNum > 0:
            testNum = testNum - 1
        # Else, decrement first order neighbor count
        else:
            firstNeigh = firstNeigh - 1
        
        # Iterate through each neighbor of current node (BFS)
        for neighbor in G.neighbors(node):
            
            # If neighbor has not been visited, put it in the queue and the list of seen nodes
            if neighbor not in seenNodes:
                Q.put(neighbor)
                seenNodes.append(neighbor)
                
                # If there are still test nodes in the queue, this was a first order neighbor
                if testNum > 0:
                    firstNeigh = firstNeigh + 1
                #else:
                   # print("The largest firstNeigh is %d" %(firstNeigh))
                    
    # The remaining queue is comprised of second order neighbors
    secondNeighbors = list(Q.queue)
    firstNeighbors = []
    testNodes = []
    
    i = 0
    while i < len(seenNodes):
        if i < (test_size + val_size):
            testNodes.append(seenNodes[i])
        else:
            if seenNodes[i] not in secondNeighbors:
                firstNeighbors.append(seenNodes[i])
        i = i+1
        
    # We now have our three lists of nodes: test, first order, second order neighbors
    # If we must remove the second neighbors entirely and chip into the first
    if num_train_keep <= len(firstNeighbors):
        keep_idx_train = np.random.choice(firstNeighbors, num_train_keep, replace=False)
        
    
    # Else, we can induce the subgraph just by removing nodes from the second neighbors
    else:
        num_keep_second = num_train_keep - len(firstNeighbors)
        keep_idx_train = np.random.choice(firstNeighbors, num_keep_second, replace=False)
        keep_idx_train = np.array(list(keep_idx_train) + firstNeighbors)
    
        
    # Trim graph and reindex
    G, id_map, class_map, feats = trim_graph(train_idx, keep_idx_train, G, id_map, class_map, feats)
    
    return G, id_map, class_map, feats


##
# Return indicies to be retained from the train set, using BFS from test set
##
def induce_rand(train_idx, G, id_map, class_map, feats):
    
    # Number of training samples to be kept
    num_train = int(len(train_idx) * FLAGS.train_percent)
    
    # Arrays of the indices of polluted nodes in each of the three sets
    keep_idx_train = np.random.choice(train_idx, num_train, replace=False)
    
    # Trim graph and reindex
    G, id_map, class_map, feats = trim_graph(train_idx, keep_idx_train, G, id_map, class_map, feats)

    return G, id_map, class_map, feats


