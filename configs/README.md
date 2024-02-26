# Config file parameters
Here are the list of customizable parameters in the config file:
```exp_id``` is used for the purposes of identifying the experiment performed.   
```load_existing``` is a flag for whether to use an existing model or not. The existing model is saved in the directory specified by results_path in the config file. False, if parameter does not exist.  
```start_epoch``` which epoch to start with. Usually used with load_existing flag.   
```exp_type``` the type of experiment to perform. IID (Independent and Identically distributed) refers to how the dataset is distributed; if it is IID. And for non-iid setting starts with non-iid setting such as ```non_iid_balanced_clients_fl```
```dset, dump_dir, dpath, seed, device_ids```are all relatively straightforward and refer to the dataset, the dump directory, dataset path, seed (for pseudorandom purposes), and the gpu ids respectively.    
```num_clients, samples_per_client``` refers to the number of collaborators (clients) participating in the learning and samples refers to the number of training images per client.    
```epochs, model_lr, batch_size``` are just the typical parameters associated with ML models.   
```alpha```  
```heterogeneous_models, models```  
```local_train_freq```  
```alpha```  
```alpha```  
```alpha```  
Others for codream/feddream:  
```data_lr``` is the lr to be used to train the data--the current (random) image that we have.  
```alpha_preds``` is the hyperparameter that is used to control the cross-entropy loss function.   
```alpha_tv``` is the hyperparameter that controls the total variation loss. Implementation of this loss function can be found in the ```modules.py``` file.      
```alpha_l2``` is the hyperparameter that applies to the l2 norm.      
```alpha_f``` is the hyperparameter applied to the r_feature, which tries to minimize the difference in the mean and the variance.   
```inp_shape``` shape of the dream dataset (here same as data image shape).
```position``` feature layer of the model from which the dream representations are extracted. If 0 dreams are extracted in the data space.
```distill_batch_size, distill_epochs``` denote the epochs and batch_size to be used for the collab learning models.  
```warmup``` refers to the amount of epochs required before you can start training on the collab data. 
```adaptive_server, adaptive_distill_start_round```  
```global_steps```  
```local_steps```  

```optimizer_type```  

Others for codream-fast/feddream-fast:cite paper, github  
```nx_samples```  
```lr_z, lr_z```  
```dset_size```  
```adv, bn, oh, bn_mmt```:fast-meta deepinversion parameters  
```reset_bn, reset_l0, ismaml```  
