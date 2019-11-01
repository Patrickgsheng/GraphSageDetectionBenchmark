from __future__ import print_function
import numpy as np
import tensorflow as tf
import json
from networkx.readwrite import json_graph as jg
import os
import induce_graph as ig
import pandas as pd
import networkx as nx
import sys
import random

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_prefix', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_100/cora', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('embedding_dir', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_100/cora_unsupervised/unsup-embedding_pre/graphsage_mean_small_0.000010', 'Directory to which the entire training embeddings are stored.')
flags.DEFINE_integer('random_seed_num', 30, 'in this experiment, we use num_train/10 by default')
flags.DEFINE_string('cleanDataset', '/Users/april/Downloads/Bart/Output/bsbm/product_100tuples_clean.csv','clean product csv file location')
flags.DEFINE_string('dirtyDataset', '/Users/april/Downloads/Bart/Output/bsbm/product_100tuples.csv','dirty product csv file location')
flags.DEFINE_string('cleanDataset2','/Users/april/Downloads/Bart/Output/bsbm/vendor_100tuples_clean.csv', 'clean vendor csv file location')
flags.DEFINE_string('dirtyDataset2','/Users/april/Downloads/Bart/Output/bsbm/vendor_100tuples.csv','dirty ')
#flags.DEFINE_string('dataset', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_50_b3/cora', 'Dataset to be used (citeseer/cora).')
flags.DEFINE_float('training_split_ratio',0.7,"the default training split for machine learning task")
flags.DEFINE_float('val_split_ratio',0.15,"the default validation split for machine learning task")
flags.DEFINE_float('test_split_ratio',0.15,"the default test split for machine learning task")
flags.DEFINE_float('pollute_ratio', 0.3, 'ratio of nodes to pollute.')
flags.DEFINE_float('attribute_pollution_ratio', 0.10, 'ratio of nodes to pollute.')
flags.DEFINE_string('destination_dir', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/BSBM/', 'Directory to which the data files will be sent.')
flags.DEFINE_string('datasetname', 'bsbm_100_error03', 'Dataset to be used (citeseer/cora).')

# Dataset and destination directory
#flags.DEFINE_string('dataset', None, 'Dataset to be used (reddit/reddit).')
#flags.DEFINE_string('destination_dir', None, 'Directory to which the data files will be sent.')
#flags.DEFINE_float('pollute_ratio', 0.2, 'ratio of nodes to pollute.')
#flags.DEFINE_float('attribute_pollution_ratio', 0.2, 'ratio of nodes to pollute.')
    

##
# Load data with specified prefix from its constituent files. Adapted from GraphSAGE's
# utils.py method 'load_data().' 
# Returns graph, ID map, class map, and feature vectors.
##


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_data(prefix, normalize=True):
    G_data = json.load(open(prefix + "-G.json"))
    G = jg.node_link_graph(G_data)
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
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():          # THROWS ERROR FOR REDDIT
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    return G, id_map, class_map, feats

##
# Load data with specified prefix from its constituent files. Adapted from GraphSAGE's
# utils.py method 'load_data().'
# Returns graph, ID map, class map, and feature vectors.
# Designed to load Bart's output dataset
##
def load_bart_data(cleanProductDataset, dirtyProductDataset, cleanVendorDataset, dirtyVendorDataset):
    cleanProductData = pd.read_csv(cleanProductDataset)
    dirtyProductData = pd.read_csv(dirtyProductDataset)
    cleanVendorData = pd.read_csv(cleanVendorDataset)
    dirtyVendorData = pd.read_csv(dirtyVendorDataset)

    #generate the unified feature length for each node (both product and vendors)
    product_vec_len = 0
    for i in range(len(cleanProductData.values[0])):

        product_vec_len += len(str(cleanProductData.values[0][i]))



    vendor_vec_len = 0
    for i in range(len(cleanVendorData.values[0])):
        if isinstance(cleanVendorData.values[0][i],str) or i==4:
            #print (cleanVendorData.values[0][i])
            vendor_vec_len += len(str(int(cleanVendorData.values[0][i])))
            #print (vendor_vec_len)
        else:
            vendor_vec_len +=1

    features = np.zeros((cleanProductData.values.shape[0]+cleanVendorData.values.shape[0], product_vec_len+vendor_vec_len))
    #reindex the node ids and fill the features
    #productData starting from 0
    #create an empty undirected graph first
    G = nx.Graph()
    num_val = 0
    num_test = 0
    #print("The num of product is used for training")
    num_product = cleanProductData.values.shape[0]
    num_train_product = int(round(num_product*FLAGS.training_split_ratio))
    num_validation_product = int(round(num_product*FLAGS.val_split_ratio))
    num_val = num_val+num_validation_product
    num_test = num_test + (num_product-num_train_product-num_validation_product)
    idx_train =[]
    idx_val =[]
    idx_test =[]
    for i in range(len(cleanProductData.values)):
        if i in range(0,num_train_product):
            #G.add_node(i,id='product'+str(cleanProductData.values[i][0]))
            G.add_node(i, key=[cleanProductData.values[i][0]])
            G.add_node(i,val=False)
            G.add_node(i,test=False)
            #by default the [1,0] indicates the clean label
            G.add_node(i,label=[1,0])
            G.add_node(i,id=i)
            G.add_node(i,feature=features[i])
            temp =[]
            idx_train.append(i)
            for t in range(len(cleanProductData.values[i])):
                if type(cleanProductData.values[i][t])==str and t>=2:
                    for x in cleanProductData.values[i][t]:
                        temp.append(x)
                if type(cleanProductData.values[i][t])==str and t<2:
                    eprint("The feature selection goes wrong")
                if type(cleanProductData.values[i][t])==int and t>=2:
                    temp_str=str(cleanProductData.values[i][t])
                    for x in temp_str:
                        temp.append(x)
                if type(cleanProductData.values[i][t])==int and t<2:
                    temp.append(str(cleanProductData.values[i][t]))
            #print (len(temp))
            temp_feature=np.zeros((1,product_vec_len))
            j=0
            while j<product_vec_len:
                for element in temp:
                    temp_feature[0][j]=float(element)
                    j+=1
            #print(G.node[i]['feature'])
            j=0
            G.node[i]['feature'][0:product_vec_len]=temp_feature[0]
            features[i] = G.node[i]['feature']
            G.node[i]['feature'] =G.node[i]['feature'].tolist()
        elif i in range(num_train_product, num_train_product+ num_validation_product):
            #print("current i is %d"%i)
            #G.add_node(i, id='product' + str(cleanProductData.values[i][0]))
            G.add_node(i, key=[cleanProductData.values[i][0]])
            G.add_node(i,val=True)
            G.add_node(i,test=False)
            G.add_node(i,label=[1,0])
            G.add_node(i,id=i)
            G.add_node(i,feature=features[i])
            temp = []
            idx_val.append(i)
            for t in range(len(cleanProductData.values[i])):
                if type(cleanProductData.values[i][t])==str and t>=2:
                    for x in cleanProductData.values[i][t]:
                        temp.append(x)
                if type(cleanProductData.values[i][t])==str and t<2:
                    eprint("The feature selection goes wrong")
                if type(cleanProductData.values[i][t])==int and t>=2:
                    temp_str=str(cleanProductData.values[i][t])
                    for x in temp_str:
                        temp.append(x)
                if type(cleanProductData.values[i][t])==int and t<2:
                    temp.append(str(cleanProductData.values[i][t]))
            temp_feature = np.zeros((1, product_vec_len))
            j = 0
            while j<product_vec_len:
                for element in temp:
                    temp_feature[0][j]=float(element)
                    j+=1
            j = 0
            G.node[i]['feature'][0:product_vec_len] = temp_feature[0]
            features[i] = G.node[i]['feature']
            G.node[i]['feature']=G.node[i]['feature'].tolist()
        else:
            #G.add_node(i, id='product' + str(cleanProductData.values[i][0]))
            G.add_node(i,key=[cleanProductData.values[i][0]])
            G.add_node(i,val=False)
            G.add_node(i,test=True)
            G.add_node(i,label=[1,0])
            G.add_node(i,id=i)
            G.add_node(i, feature=features[i])
            temp = []
            idx_test.append(i)
            for t in range(len(cleanProductData.values[i])):
                if type(cleanProductData.values[i][t])==str and t>=2:
                    for x in cleanProductData.values[i][t]:
                        temp.append(x)
                if type(cleanProductData.values[i][t])==str and t<2:
                    eprint("The feature selection goes wrong")
                if type(cleanProductData.values[i][t])==int and t>=2:
                    temp_str=str(cleanProductData.values[i][t])
                    for x in temp_str:
                        temp.append(x)
                if type(cleanProductData.values[i][t])==int and t<2:
                    temp.append(str(cleanProductData.values[i][t]))
            temp_feature = np.zeros((1, product_vec_len))
            j = 0
            while j<product_vec_len:
                for element in temp:
                    temp_feature[0][j]=float(element)
                    j+=1
            j = 0
            G.node[i]['feature'][0:product_vec_len] = temp_feature[0]
            features[i] = G.node[i]['feature']
            G.node[i]['feature'] =G.node[i]['feature'].tolist()
    product_num = len(G.node)
    num_vendor_product = cleanVendorData.values.shape[0]
    num_train_vendor = int(round(num_vendor_product * FLAGS.training_split_ratio))
    num_validation_vendor = int(round(num_vendor_product * FLAGS.val_split_ratio))
    num_val = num_val + num_validation_vendor
    num_test = num_test + (num_vendor_product - num_train_vendor - num_validation_vendor)
    for i in range(product_num,product_num+cleanVendorData.values.shape[0]):
        if i in range(product_num, product_num+num_train_vendor):
            '''
            G.add_node(i,id='product'+str(cleanVendorData.values[i-product_num][3])+'vendor'\
                       +str(cleanVendorData.values[i-product_num][1])+'offer'\
                       +str(cleanVendorData.values[i-product_num][2]))
            '''
            G.add_node(i,key=[cleanVendorData.values[i-product_num][3],cleanVendorData.values[i-product_num][1],cleanVendorData.values[i-product_num][2]])
            G.add_node(i,val=False)
            G.add_node(i,test=False)
            G.add_node(i,label=[1,0])
            G.add_node(i, id=i)
            G.add_node(i, feature=features[i])
            temp = []
            idx_train.append(i)
            for t in range(len(cleanVendorData.values[i-product_num])):
                if t==4:
                    #cleanVendorData.values[i - product_num].astype(int)
                    temp_str= str(cleanVendorData.values[i-product_num][t].astype(int))
                    for x in temp_str:
                        temp.append(x)
                else:
                    temp.append(str(cleanVendorData.values[i-product_num][t].astype(int)))
            temp_feature = np.zeros((1, vendor_vec_len))
            j = 0
            while j<vendor_vec_len:
                for element in temp:
                    temp_feature[0][j]=float(element)
                    j+=1
            j = 0
            G.node[i]['feature'][product_vec_len:product_vec_len+vendor_vec_len] = temp_feature[0]
            features[i] = G.node[i]['feature']
            G.node[i]['feature'] =G.node[i]['feature'].tolist()
        elif i in range(product_num+num_train_vendor, product_num+num_train_vendor+num_validation_vendor):
            '''
            G.add_node(i, id='product' + str(cleanVendorData.values[i - product_num][3]) + 'vendor' \
                             + str(cleanVendorData.values[i - product_num][1]) + 'offer' \
                             + str(cleanVendorData.values[i - product_num][2]))
            '''
            G.add_node(i, key=[cleanVendorData.values[i - product_num][3], cleanVendorData.values[i - product_num][1],
                               cleanVendorData.values[i - product_num][2]])
            G.add_node(i, val=True)
            G.add_node(i, test=False)
            G.add_node(i, label=[1,0])
            G.add_node(i, id=i)
            G.add_node(i, feature=features[i])
            temp = []
            idx_val.append(i)
            for t in range(len(cleanVendorData.values[i-product_num])):
                if t==4:
                    #cleanVendorData.values[i - product_num].astype(int)
                    temp_str= str(cleanVendorData.values[i-product_num][t].astype(int))
                    for x in temp_str:
                        temp.append(x)
                else:
                    temp.append(str(cleanVendorData.values[i-product_num][t].astype(int)))
            temp_feature = np.zeros((1, vendor_vec_len))
            j = 0
            while j < vendor_vec_len:
                for element in temp:
                    temp_feature[0][j] = float(element)
                    j += 1
            j = 0
            G.node[i]['feature'][product_vec_len:product_vec_len + vendor_vec_len] = temp_feature[0]
            features[i] = G.node[i]['feature']
            G.node[i]['feature'] =G.node[i]['feature'].tolist()
        else:
            '''
            G.add_node(i, id='product' + str(cleanVendorData.values[i - product_num][3]) + 'vendor' \
                             + str(cleanVendorData.values[i - product_num][1]) + 'offer' \
                             + str(cleanVendorData.values[i - product_num][2]))
            '''
            G.add_node(i, key=[cleanVendorData.values[i - product_num][3], cleanVendorData.values[i - product_num][1],
                               cleanVendorData.values[i - product_num][2]])
            G.add_node(i,val=False)
            G.add_node(i,test=True)
            G.add_node(i, label=[1, 0])
            G.add_node(i, id=i)
            G.add_node(i, feature=features[i])
            temp = []
            idx_test.append(i)
            for t in range(len(cleanVendorData.values[i-product_num])):
                if t==4:
                    #cleanVendorData.values[i - product_num].astype(int)
                    temp_str= str(cleanVendorData.values[i-product_num][t].astype(int))
                    for x in temp_str:
                        temp.append(x)
                else:
                    temp.append(str(cleanVendorData.values[i-product_num][t].astype(int)))
            temp_feature = np.zeros((1, vendor_vec_len))
            j = 0
            while j < vendor_vec_len:
                for element in temp:
                    temp_feature[0][j] = float(element)
                    j += 1
            j = 0
            G.node[i]['feature'][product_vec_len:product_vec_len + vendor_vec_len] = temp_feature[0]
            features[i] = G.node[i]['feature']
            G.node[i]['feature'] =G.node[i]['feature'].tolist()
    #Scan the graph vendor list and construct the edgeList
    print("Scan the graph vendor list and construct the edgeList with product entity")
    vendor_dict= dict()
    for i in range(product_num, product_num + cleanVendorData.values.shape[0]):
        #product_id starting from 0
        product_id= int(G.node[i]['key'][0]-1)
        vendor_product_id = str(G.node[i]['key'][1])+str(G.node[i]['key'][0])
        G.add_edge(i,product_id)
        if not vendor_dict.has_key(vendor_product_id):
            vendor_dict[vendor_product_id]=[]
        vendor_dict[vendor_product_id].append(i)
    print("from vendor entity to product entity the edgeList is built")
    #scan the vendor entity and connect those entities with the same vendor id
    print("Scan the vendor entity and connect those entities with the same vendor id")
    for key in vendor_dict.keys():
        for node1 in vendor_dict[key]:
            for node2 in vendor_dict[key]:
                if node1!=node2:
                    #print(node1)
                    #print(node2)
                    G.add_edge(node1, node2)

    print("edgeList is built and done !!!")
    #scan the clean tables and get the feats


    # Create id-Map and class-Map
    num_nodes = G.number_of_nodes()
    print("Creating id-Map and class-Map...")
    # Initialize the dictionarys
    idM = {}
    classM = {}

    # Populate the dictionarys
    i = 0
    while i < num_nodes:
        idStr = str(i)
        idM[idStr] = i
        classM[idStr] = list(G.node[i]['label'])
        i += 1
    return G, idM, classM, features,num_val, num_test,idx_train, idx_val, idx_test





    #scan the two tables, identify dirty nodes in the dirtyDataset and label them






##
# Pollutes graph, randomly selecting features from randomly selected nodes. The pollute ratio and
# attribute pollution ratio flags determine the probability that a given node or attribute will
# be corrupted, respectively. Also labels node accordingly, in graph and class map.
# Returns graph, class map, and feature vectors.
##
def pollute_graph(G, idMap, classMap, feats, num_val, num_test,idx_train, idx_val, idx_test):
    print ("Polluting data\n")
    
    # Number of nodes, number of nodes in validation and test sets
    num_nodes = G.number_of_nodes()
    
    # Number of polluted nodes in train, validationm, and test sets, respectively
    poll_num_train = int((num_nodes - (num_val+num_test)) * FLAGS.pollute_ratio)
    poll_num_val = int(num_val * FLAGS.pollute_ratio)
    poll_num_test = int(num_test * FLAGS.pollute_ratio)
    
    '''
    # Index of first validation node and first test node, respectively
    idx_val = (num_nodes - 1) - (num_val + num_test)
    idx_test = (num_nodes - 1) - (num_test)
    '''
    
    # Arrays of the indices of polluted nodes in each of the three sets
    poll_idx_train = np.random.choice(idx_train, poll_num_train, replace=False)
    poll_idx_val = np.random.choice(idx_val, poll_num_val, replace=False)
    poll_idx_test = np.random.choice(idx_test, poll_num_test, replace=False)
    
    # The number of attributes and polluted attributes in the feature vector of a node
    attr_dim = len(G.node[0]['feature'])
    poll_num_attr = int(attr_dim * FLAGS.attribute_pollution_ratio)
    
    # Iterate through each node in the graph
    for n in list(G):
        
        # Assign to train, val, or test set
        if n in idx_train:
            if G.node[n]['val'] != False:
                eprint("Train instance %d goes wrong!!!" %n)
            if G.node[n]['test'] != False:
                eprint("Train instance %d goes wrong!!!" %n)


        elif n in idx_val:
            if G.node[n]['val'] != True:
                eprint("Validation instance %d goes wrong!!!" %n)
            if G.node[n]['test'] != False:
                eprint("Validation instance %d goes wrong!!!" % n)
        elif n in idx_test:
            if G.node[n]['val'] != False:
                eprint("Test instance %d goes wrong!!!" % n)
            if G.node[n]['test'] != True:
                eprint("Test instance %d goes wrong!!!" % n)
        
        # If the node is to be polluted, proceed to its features
        if (n in poll_idx_train) or (n in poll_idx_val) or (n in poll_idx_test):
            G.node[n]['label'] = [0, 1]
            
            poll_attr = np.random.choice(attr_dim, poll_num_attr, replace=False)
            
            # Iterate through each of the node's features
            i = 0
            while i < poll_num_attr:
                
                # If this feature is polluted, switch its value
                if G.node[n]['feature'][poll_attr[i]] == 1.0:
                    #print(G.node[n]['feature'][poll_attr[i]])
                    G.node[n]['feature'][poll_attr[i]] = 0.0
                    feats[n][poll_attr[i]] = 0.0
                elif G.node[n]['feature'][poll_attr[i]] == 0.0:
                    G.node[n]['feature'][poll_attr[i]] = 1.0
                    feats[n][poll_attr[i]] = 1.0
                else:
                    G.node[n]['feature'][poll_attr[i]] = random.randrange(feats.shape[0])
                    feats[n][poll_attr[i]] = random.randrange(feats.shape[0])
                i += 1
            
            classMap[str(n)] = [0, 1]
        
        # Else, label it as clean
        else:
            G.node[n]['label'] = [1, 0]
            classMap[str(n)] = [1, 0]
    '''    
    # Delete original labels
    for n in list(G):
        del G.node[n]['labels']
    '''
        
    return idx_train, G, classMap, feats

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
    
    # Load data
    #G, idMap, classMap, feats = load_data(FLAGS.dataset)
    G,idMap,classMap, feats,num_val, num_test, idx_train, idx_val, idx_test\
        =load_bart_data(FLAGS.cleanDataset, FLAGS.dirtyDataset, FLAGS.cleanDataset2, FLAGS.dirtyDataset2)


    
    # Pollute graphs
    trainIdx, G, classMap, feats = pollute_graph(G, idMap, classMap, feats,num_val, num_test,\
                                                 idx_train, idx_val, idx_test)
    
    # Induce Graph
    G, idMap, classMap, feats = ig.induce_rand(trainIdx, G, idMap, classMap, feats)
    
    datasetName = FLAGS.datasetname.split("/")[-1]
    
    # Dump everything into .json files and one .npy
    dumpJSON(FLAGS.destination_dir, datasetName, G, idMap, classMap, feats)
    
    
if __name__ == "__main__":
    main()
