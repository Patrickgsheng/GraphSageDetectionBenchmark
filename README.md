# GraphSAGE Benchmark

### Preprocessing

##### Changing Formats

The first step is to turn the specified dataset into the format required by the GraphSAGE algorithm. GraphSAGE takes, as input, a *Networkx* graph, a classification map, and an index map, all stored as .json files. It also requires a feature matrix, stored as a .npy. 

In order to convert the existing dataset into this format, *cd* into the "Preprocessing" directory. Here, you will find the scripts, as well as a sub-directory containing the datasets. In order to run the program, use the following command:

```bash
$ python prep_data_gs.py --dataset MY_DSET --destination_dir MY_DIRECTORY
```

The two flags above must be set. ```—dataset``` will be either "citeseer" or "cora", and ```—destination_dir``` will be whatever sub-directory in which you'd like to store the output. 

You can, optionally, set the ```—pollute_ratio``` and ```—attribute_pollution_ratio``` flags. These will be floats between 0 and 1, determining the percentage of training, validation, and test samples that will be dirtied and the percentage of attributes within these nodes that will be corrupted. The default is 20% (0.2).

Once the program has been run, your sub-directory should contain the three *json* files and the one *numpy* file. In order to proceed to the next step, this subdirectory must be moved/copied to the "GraphSAGE/graphsage" directory.

##### GraphSAGE-ready Pollution

Some datasets may already be in the format accepted by GraphSAGE. In order to pollute and label them, use the following command:

```bash
$ python prep_preformatted.py --dataset MY_DSET --destination_dir MY_DIRECTORY
```

In this case, ```—dataset``` must be set to the path of the dataset, as well as its prefix. For example, if you wish to pollute the *toy-ppi* dataset, contained in the subdirectory "example_data," comprised of the files "toy-ppi-G.json," "toy-ppi-id_map.json," etc, then you would set ```—dataset example_data/toy-ppi```. The flags for attribute and node pollution are identical to the previous program. 

### GraphSAGE

In order to run GraphSAGE, you must have Docker installed. Once that is done, the program can be built and run with the following (lifted from their README):

```bash
# build like so:
$ docker build -t graphsage .

# once it's built, use the command below to run. This will open up graphsage's bash 
$ docker run -it graphsage bash
```

This will build and open up GraphSAGE's bash. Now, you must *cd* into the "graphsage" sub-directory, where your dataset sub-directory should already be stored. Once there, you can train the supervised model:

```bash
$ python supervised_train.py --train_prefix EXAMPLE_DIRECTORY/MY_DATA --model MY_MODEL --base_log_dir RESULT_DIRECTORY --batch_size SIZE
```

The ```—train_prefix``` flag must be set. For example, if the "cora" dataset is stored in the directory "cora_data," then it will be "cora_data/cora." The model is set to "graphsage_mean," by default, but can be changed to any of the six aggregators. The ```—base_log_dir``` specifies the location of the output; in the case of the supervised model, it will store the F1 scores (working on precision and recall), as well as the elapsed time. Last, when run on not-super-computers, ```—batch_size``` can be changed (the default is 512). More information can be found in the README for the project (in the GraphSAGE folder).