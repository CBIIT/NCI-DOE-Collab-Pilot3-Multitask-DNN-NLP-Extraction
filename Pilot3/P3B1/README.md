## Model Description

This resource trains a multitask DNN using Keras, a deep learning library, on a given sample corpus of biomedical text (i.e. pathology reports). The trained model can then be used to extract multiple pieces of information such as tumor sites, tumor laterality, and tumor grade from pathology reports. 

The network itself is composed of layers shared among multiple tasks and individual layers specific for each task as shown in the graphical representation of the model architecture shown below:

<img src="https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/multilayer-dnn.png" width="400" height="400" alt="Multilayer DNN Architecture">

The user can modify the parameters of the model in [p3b1_default_model.txt](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1_default_model.txt). 

Some of the parameters the user can adjust include: 
 * number of iterations for training the model
 * number of layers and nodes in the shared layers
 * number of layers and nodes in the individual layers
 * learning rate
 * batch size
 * task names (i.e. tumor sites)
 * number of folds for cross-validation

Descriptions of the parameters are included in [p3b1.py](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1.py).

## Setup

To set up the Python environment needed to train and run this model:
1. Install [conda](https://docs.conda.io/en/latest/) package manager. 
2. Clone this repository. 
3. Create the environment as shown below.

```bash
   conda env create -f environment.yml -n P3B1
   conda activate P3B1
```

To download the preprocessed data needed to train and test the model:
1. Create an account on the Model and Data Clearinghouse [MoDaC](https://modac.cancer.gov). 
2. Follow the instructions in the Training section below.
3. When prompted by the training script, enter your MoDaC credentials.

## Training

(Optional) Modify the parameters of the model in [p3b1_default_model.txt](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1_default_model.txt). 

   |	Parameter	|	Description	|
   |	-------------	|	-------------	|
   | shared_nnet_spec | Network structure of shared layer |
   | ind_nnet_spec | Network structure of task-specific layer|
   | batch_size	| Number of samples that will be passed through to the network at one time (int) |
   | epochs | Number of training iterations (int) |
   | learning_rate | Learning rate (float) |
   | dropout | Fraction of units to drop for the linear transformation of the inputs (float)|
   | loss | Loss function to use |
   | activation | Activation function to use |
   | out_activation | Activation function to use for the output layer |
   | optimizer | Optimizer method to use |
   | metrics | Metric function to use |
   | feature_names | Name of the tasks |
   | n_fold | Number of folds for cross-validation (int)|
   | timeout | Duration of time for model training in seconds (int)|
   | output_dir | Name of the folder for the output files |


To train the model from scratch, run [p3b1_baseline_keras2.py](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1_baseline_keras2.py). 

```
python p3b1_baseline_keras2.py
```

This script does the following:
 * Downloads and uncompresses the preprocessed data file from MoDaC.
 * Prepares the training and testing data.
 * Builds the multitask DNN model.
 * Trains the model and logs each iteration in a JSON file.
 * Prints the performance metrics of the model for each task.

### Example output
```
Task 1 : Primary site - Macro F1 score 0.15679785330948118
Task 1 : Primary site - Micro F1 score 0.3838383838383838
Task 2 : Tumor laterality - Macro F1 score 0.6312178387650085
Task 2 : Tumor laterality - Micro F1 score 0.6511627906976745
Task 3 : Histological grade - Macro F1 score 0.2923280423280423
Task 3 : Histological grade - Micro F1 score 0.4852941176470588
Average loss:  1.1945778330167134
```

Note that the training and testing data files are provided as standard CSV files in *Data/Pilot3/P3B1_data*. We do not make any representations or guarantees about the data. It is provided solely as a starter set to facilitate end to end training of a model. The accuracy and results of that model are not to be strongly considered.

## Acknowledgments
   
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
