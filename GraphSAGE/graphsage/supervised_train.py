from __future__ import division
from __future__ import print_function

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

# NOTE: All of these were originally graphsage.blah...
from supervised_models import SupervisedGraphsage
from models import SAGEInfo
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.005, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/BSBM/bsbm_100_error03', 'prefix identifying training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 35, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 43, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 16, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
#according to the readMe file, set to a positive number
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/BSBM/bsbm_100_error04', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 315, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 10, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def calc_scores(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        #print (y_true[0])
        #print (y_pred[0])
        #Scan all the y_true, check how many of them are 1s
        '''
        count = 0
        for y in y_true:
            if y==1:
                count+=1
        #print("The total error is %d"%count)
        '''

    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_accuracy_score=metrics.accuracy_score(y_true, y_pred,normalize=True)

    #print("accuracy is {:.5f}".format(balanced_accuracy_score))
    return precision[-1], recall[-1], fscore[-1], support[-1],balanced_accuracy_score
# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=feed_dict_val)
    #print("label's shape in evaluation mode %s"%str(labels.shape))
    #print("output's shape in evaluation mode %s"%str(node_outs_val[0].shape))
    precision, recall, fscore, support,accuracy = calc_scores(labels, node_outs_val[0])
    return node_outs_val[1], precision, recall, fscore, support, accuracy,(time.time() - t_test)

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    print("\\\\\\\\\\\\\\\\\\\\\Incremental evaluation starts:\\\\\\\\\\\\\\\\\\\\\\\\")
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    #print("In the incremental_evaluate the val_preds shape is %s" %str(val_preds.shape))
    #print("In the incremental_evaluate the labels shape is %s" % str(labels.shape))
    scores = calc_scores(labels, val_preds)
    return labels, val_preds, np.mean(val_losses), scores[0], scores[1], scores[2], scores[3], scores[4],(time.time() - t_test)

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, test_data=None):

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    #context_pairs in the default setting is []
    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos, 
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    # Train model
    
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            
            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_precision, val_recall, val_f1, val_support, val_accuracy, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                else:
                    val_cost, val_precision, val_recall, val_f1, val_support, val_accuracy, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)

                labels_test, preds_test, test_cost, test_precision, test_recall, test_f1, test_support, test_accuracy, duration = incremental_evaluate(
                    sess, model, minibatch, FLAGS.batch_size, test=True)


                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost



            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_precision, train_recall, train_f1,train_support,train_accuracy = calc_scores(labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_precision=", "{:.5f}".format(train_precision), 
                      #"train_recall=", "{:.5f}".format(train_recall),
                      #"train_f1=", "{:.5f}".format(train_f1),
                      "train_accuracy=",'{:.2f}'.format(train_accuracy),
                      #"train_support=", "{:.5f}".format(train_support),
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_precision=", "{:.5f}".format(val_precision),
                      "val_accuracy=",'{:.2f}'.format(val_accuracy),
                      #"val_recall=", "{:.5f}".format(val_recall),
                      #"val_f1=", "{:.5f}".format(val_f1),
                      #"val_support=", "{:.5f}".format(val_support),
                       "test_loss=","{:.5f}".format(test_cost),
                      "test_precision=", "{:.5f}".format(test_precision),
                      "test_accuracy=",'{:.2f}'.format(test_accuracy),
                      "test_recall=", "{:.5f}".format(test_recall),
                      "test_f1=", "{:.5f}".format(test_f1),
                      "time=", "{:.5f}".format(avg_time))
 
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break
    
    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    labels, preds, val_cost, val_precision, val_recall, val_f1, val_support, val_accuracy,duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
    print("Full validation stats:",
                  "loss=", "{:.5f}".format(val_cost),
                  "precision=", "{:.5f}".format(val_precision),
                  "recall=", "{:.5f}".format(val_recall),
                  "f1=", "{:.5f}".format(val_f1),
                  "accuracy=","{:.2f}".format(val_accuracy),
                  "support=", "{:.5f}".format(val_support),
                  "time=", "{:.5f}".format(duration))
    with open(log_dir() + "val_stats.txt", "w") as fp:
        fp.write("loss={:.5f} precision={:.5f} recall={:.5f} f1={:.5f} support={:.5f} time={:.5f}".
                format(val_cost, val_precision, val_recall, val_f1, val_accuracy, val_support, duration))

    print("Writing test set stats to file (don't peak!)")
    labels_test, preds_test, test_cost, test_precision, test_recall, test_f1, test_support, test_accuracy,duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True)
    print("Full testing stats:",
          "loss=", "{:.5f}".format(test_cost),
          "precision=", "{:.5f}".format(test_precision),
          "recall=", "{:.5f}".format(test_recall),
          "f1=", "{:.5f}".format(test_f1),
           "accuracy=","{:.2f}".format(test_accuracy),
          "support=", "{:.5f}".format(test_support),
          "time=", "{:.5f}".format(duration))
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} precision={:.5f} recall={:.5f} f1={:.5f} support={:.5f}".
                format(test_cost, test_precision, test_recall, test_f1, test_accuracy, test_support))

def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    
    train(train_data)
    
    '''
    # If no model is chosen, loop through each
    if FLAGS.model == None:
        FLAGS.model = 'graphsage_mean'
        i = 0
        
        while i < 5:
            if FLAGS.model == 'graphsage_mean':
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MEAN-BASED~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Loading training data..")
                train_data = load_data(FLAGS.train_prefix)
                print("Done loading training data..")
                train(train_data)
                FLAGS.model = 'gcn'
                
            elif FLAGS.model == 'gcn':
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GCN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Loading training data..")
                train_data = load_data(FLAGS.train_prefix)
                print("Done loading training data..")
                train(train_data)
                FLAGS.model = 'graphsage_seq'
                
            elif FLAGS.model == 'graphsage_seq':
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LSTM-BASED~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Loading training data..")
                train_data = load_data(FLAGS.train_prefix)
                print("Done loading training data..")
                train(train_data)
                FLAGS.model = 'graphsage_meanpool'
                
            elif FLAGS.model == 'graphsage_meanpool':
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MEAN-POOL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Loading training data..")
                train_data = load_data(FLAGS.train_prefix)
                print("Done loading training data..")
                train(train_data)
                FLAGS.model = 'graphsage_maxpool'
                
            elif FLAGS.model == 'graphsage_maxpool':
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAX-POOL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Loading training data..")
                train_data = load_data(FLAGS.train_prefix)
                print("Done loading training data..")
                train(train_data)
            
            i += 1
    else:
        train(train_data)
        
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    labels_sam = np.array([[0, 1],
                   [0, 1],
                   [1, 0],
                   [1, 0],
                   [1, 0]], dtype=np.float32)
    logits_sam = np.array([[0, 1],
                   [1, 0],
                   [1, 0],
                   [1, 0],
                   [1, 0]], dtype=np.float32)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(labels_sam, logits_sam, average=None)
    print(precision)
    print(recall)
    sess.run(labels)
    '''

if __name__ == '__main__':
    tf.app.run()
