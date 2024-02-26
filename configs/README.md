# Config file parameters
Here is the list of customizable parameters in the config file:  
* ```exp_id``` is used to identify the experiment performed.   
* ```load_existing``` is a flag for whether to use an existing model or not. The existing model is saved in the directory specified by results_path in the config file. False, if the parameter does not exist.  
* ```start_epoch``` which epoch to start with. Usually used with load_existing flag.   
* ```exp_type```is the type of experiment to perform. IID (Independent and Identically distributed) refers to how the dataset is distributed; if it is IID. And for non-iid setting starts with a non-iid setting such as ```non_iid_balanced_clients_fl```.  
* ```dset, dump_dir, dpath, seed, device_ids```are all relatively straightforward and refer to the dataset, the dump directory, dataset path, seed (for pseudorandom purposes), and the GPU ids respectively.    
* ```num_clients, samples_per_client``` refers to the number of collaborators (clients) participating in the learning and samples refer to the number of training images per client.    
* ```epochs, model_lr, batch_size``` are just the typical parameters associated with ML models.   
* ```alpha``` used to indicate the degree of data heterogeneity for non-iid setting, lower alpha indicates highly non-id data  
* ```heterogeneous_models, models```  is a flag used to indicate if clients have heterogeneous models. If True, then the list of different models for each client is provided
Others for Codream/feddream:  
* ```data_lr``` is the lr to be used to train the data--the current (random) image that we have.  
* ```alpha_preds``` is the hyperparameter that is used to control the cross-entropy loss function.   
* ```alpha_tv``` is the hyperparameter that controls the total variation loss. Implementation of this loss function can be found in the ```modules.py``` file.      
* ```alpha_l2``` is the hyperparameter that applies to the l2 norm.      
* ```alpha_f``` is the hyperparameter applied to the r_feature, which tries to minimize the difference in the mean and the variance.   
* ```inp_shape``` shape of the dream dataset (here same as data image shape).
* ```position``` feature layer of the model from which the dream representations are extracted. If 0 dreams are extracted in the data space.
* ```distill_batch_size, distill_epochs``` denote the epochs and batch_size to be used for the collab learning models.  
* ```warmup``` refers to the amount of epochs required before you can start training on the collab data.  
* ```adaptive_server, adaptive_distill_start_round``` is a flag used to indicate if we are doing the adaptive training of collab-dreams w.r.t. server or not. We indicate when this adaptive training starts with start_round parameter.
* ```local_steps``` is the number of local steps of dreaming on each client's local model in each epoch.  
* ```global_steps``` is the number of global steps for aggregation of dreams by the server in each epoch.  
* ```optimizer_type``` is the type of aggregation performed for accumulating knowledge.   

Others for codream-fast/feddream-fast:
* ```nx_samples``` is the number of batches of dreams generated in each epoch.  
* ```lr_z, lr_g```  is the learning rate of meta-intitialization(z) and generator(g).  
* ```dset_size``` is the size of the dream dataset buffer. As training progresses and dreams improve we keep discarding old dreams in a sliding window manner of the size of the dataset. It is represented as (number of batches of dreams * batch-size).  
* ```adv, bn, oh, bn_mmt``` are fast-meta deepinversion parameters.    
* ```reset_bn, reset_l0, ismaml```  are the choices of optimizers for fast-meta deepinversion.
