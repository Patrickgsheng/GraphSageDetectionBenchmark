import numpy as np
import tensorflow as tf
import random
from scipy.sparse import csr_matrix

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('pollute_ratio', 0.1, 'ratio of nodes to pollute.')
flags.DEFINE_float('attribute_pollution_ratio', 0.02, 'ratio of nodes to pollute.')

def pollute_data(labels,attributes, idx_train, idx_val, idx_test):
    print ("Polluting data\n")
    # transform attributes to csr format
    attributes = attributes.tocsr()
    #Get the shape of label
    x = attributes.shape[0]
    #We only have dirty/clean instances
    y=2
    y_train = np.zeros((x, y))
    y_val = np.zeros((x, y))
    y_test = np.zeros((x, y))
    

    # pollute_train
    # Get the random_pollute_train_instance from training dataset
    pollute_train_size = int(len(idx_train) * FLAGS.pollute_ratio)
    pollute_train_row = np.random.choice(idx_train, pollute_train_size, replace=False)
    pollute_val_size = int(len(idx_val) * FLAGS.pollute_ratio)
    pollute_val_row = np.random.choice(idx_val, pollute_val_size, replace=False)
    pollute_test_size = int(len(idx_test) * FLAGS.pollute_ratio)
    pollute_test_row = np.random.choice(idx_test, pollute_test_size, replace=False)
    

    temp_count =0

    for i in idx_train:
        if i in pollute_train_row:
            #col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            print("polluting",i)
            col_pollute = random.sample(range(attributes.shape[1]),int(attributes.shape[1] * FLAGS.attribute_pollution_ratio))
            col_pollute = np.array(col_pollute)
            col_pollute.reshape(int(attributes.shape[1] * FLAGS.attribute_pollution_ratio),1)
            #print (col_pollute)
            #Always use the second bit to indicate dirty instance
            y_train[i][1]= 1
            for j in col_pollute:
                if attributes[i,j]==0:
                    attributes[i,j]=1
                else:
                    attributes[i,j]=0
                print("pollute training data")
                print(i, j, attributes[i,j], temp_count)
            temp_count+=1
        else:
            y_train[i][0] = 1
            print(i)
    for i in idx_val:
        if i in pollute_val_row:
            #col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            col_pollute = random.sample(range(attributes.shape[1]),
                                        int(attributes.shape[1] * FLAGS.attribute_pollution_ratio))
            col_pollute = np.array(col_pollute)
            col_pollute.reshape(int(attributes.shape[1] * FLAGS.attribute_pollution_ratio), 1)
            y_val[i][1] = 1
            for j in col_pollute:
                if attributes[i,j]==0:
                    attributes[i,j]=1
                else:
                    attributes[i,j]=0
                print("pollute validation data")
                print(i, j, attributes[i, j], temp_count)
            temp_count += 1
        else:
            y_val[i][0]= 1
            #print(i)
    for i in idx_test:
        if i in pollute_test_row:
            #col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            col_pollute = random.sample(range(attributes.shape[1]),
                                        int(attributes.shape[1] * FLAGS.attribute_pollution_ratio))
            col_pollute = np.array(col_pollute)
            col_pollute.reshape(int(attributes.shape[1] * FLAGS.attribute_pollution_ratio), 1)
            y_test[i][1] = 1
            for j in col_pollute:
                if attributes[i,j]==0:
                    attributes[i,j]=1
                else:
                    attributes[i,j]=0
                print("pollute testing data")
                print(i, j, attributes[i, j], temp_count)
            temp_count += 1
        else:
            y_test[i][0]= 1
            #print(i)

    labels = np.zeros((x, y))
    labels = y_train + y_val + y_test
    '''
    #save processed data
    print("Saving the processed polluted features and labels")
    attributes_np = csr_matrix(attributes).toarray()
    np.savetxt('/Users/april/Downloads/GraphSAGE_Benchmark-master/Preprocessing/data/attributes_processed',attributes_np,delimiter=",")
    np.savetxt('/Users/april/Downloads/GraphSAGE_Benchmark-master/Preprocessing/data/training_label',y_train,delimiter=",")
    np.savetxt('/Users/april/Downloads/GraphSAGE_Benchmark-master/Preprocessing/data/validation_label',y_val,delimiter=',')
    np.savetxt('/Users/april/Downloads/GraphSAGE_Benchmark-master/Preprocessing/data/testing_label',y_test,delimiter=',')
    '''

    # print labels.shape
    return attributes, labels