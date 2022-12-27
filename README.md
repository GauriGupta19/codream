# Collab Learning
Collaborative learning by sharing distilled images

# Control Flow

## Main
The application first runs main.py, which handles running the application from the command line. It takes as argument a config file path, where the default config file is ```iid_clients.py```.  

The usage for main.py is as follows:  
``` python3 main.py -b "{path_to_config_file}" ```, where you replace the {path_to_config_file} with the actual path of the desired config file.

Upon launch, ```main.py``` parses the config file, and based on its configuration runs Scheduler.py, which takes care of the rest of our program.

### Config file parameters
```exp_id``` is used for the purposes of identifying the experiment performed.   
```load_existing``` is a flag for whether to use an existing model or not. The existing model is saved in the directory specified by results_path in the config file. False, if parameter does not exist.  
```start_epoch``` which epoch to start with. Usually used with load_existing flag.   
```exp_type``` the type of experiment to perform. Isolated and collab determine whether to perform the tests in isolation or with the data of other clients (collab). IID (Independent and Identically distributed) refers to how the dataset is distributed; if it is IID, then there's training that's disjoint from other classes. Otherwise, there may be portions of training and testing that overlap between classes.  
```dset, dump_dir, dpath, seed, device_ids```are all relatively straightforward and refer to the dataset, the dump directory, dataset path, seed (for pseudorandom purposes), and the gpu ids respectively.    
```num_clients, samples_per_client``` refers to the number of collaborators (clients) participating in the learning and samples refers to the number of training images per client.    
```epochs, model_lr, batch_size``` are just the typical parameters associated with ML models.   
```data_lr``` is the lr to be used to train the data--the current (random) image that we have.  
```alpha_preds``` is the hyperparameter that is used to control the cross-entropy loss function.   
```alpha_tv``` is the hyperparameter that controls the total variation loss. Implementation of this loss function can be found in the ```modules.py``` file.      
```alpha_l2``` is the hyperparameter that applies to the l2 norm.      
```alpha_f``` is the hyperparameter applied to the r_feature, which tries to minimize the difference in the mean and the variance.   
```inp_shape, position``` describe the model we're making.  
```distill_batch_size, distill_epochs``` denote the epochs and batch_size to be used for the collab learning models.  
```warmup``` refers to the amount of epochs required before you can start training on the collab data.  

## Scheduler
Scheduler takes care of the bulk of our program. Mainly, it executes any one of the iid/non-iid client programs in isolation or in collaboration with other clients, depending on the parameters given in ```exp_type```. Here,  I will briefly walk through each of these training methods and how they differ:

#### 1) IID Clients Isolated
This performs a typical training that we see in machine learning. For epochs specified in the config file, it tests each of the client's model and logs the test results. Then, it trains each of the client's model using Cross Entropy loss and adjusts the parameters to better fit the data. It also saves the model by client numbers in the results path specified in the config file. All of this is done from start epoch to epochs as indicated by the config file. 

#### 2) IID Federated
This is similar to (1), but with a major difference. After each epoch is done, following the training, the parameters for each of the client's model is set to be the weighted average among all the client models.

#### 3) IID Clients Collab
The structure of this training is very similar to (1), but there are important distinctions. We still have the same sequence of test, train, save for epochs from start epochs to epochs. However, once you're done with the "warmup" number of epochs, as described in the config, then you're going to generate random noise for your image and generate a random label. You apply gradient ascent--this is more of the Deep Inversion technique detailed by the paper linked below--on the image and label to get a generated image of the label. Then you run the image through the model to get an appropriate activation according to the model and store that. You store this collab data for all the models. Then, you train each client on the collab data and its own data, using the KL loss function. You repeat this for epochs as specified in the config and you continue adding to the collab data (without clearing it) as you go into further epochs. 

#### 4) IID Clients Adaptive Collab
This is similar to (3), but with some differences that we'll detail here. For each client, instead of applying gradient ascent subject to its own loss only, in adaptive collab, we apply gradient ascent to a client's model subject to the loss from all other clients and loss on the client model itself as well. This also results in a slightly different regularization of hyperparameters as can be seen in the ```algos.py``` file. 

#### 5) Non-IID Clients Colab
This is the same as clients colab, but the dataset is not mutually exclusive anymore (non-iid). In addition, the training method is also different as we use the loss from both KL Loss and Cross Entropy to adjust our parameters. 

#### 6) Non-IID Federated
This is also the same as the IID Federated method, with the exception that the datasets now are not necessarily independent or identically distributed.







To further examine these training methods, take a look at the research paper on which this project is based in [here](https://arxiv.org/abs/1912.08795).  




