## Model Description

The RNN-LSTM based Clinical Text Generator trains a Long Short-Term Memory, or LSTM, recurrent neural network, using Keras, a deep learning library, on a given sample corpus of biomedical text (i.e. pathology reports). The trained model can then be used to synthesize text documents similar in context to the sample corpus.

The user can modify the parameters of the model in [p3b1_default_model.txt](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1_default_model.txt). Some of the parameters the user can adjust include: 
 * number of iterations for training the model
 * number of layers in the LSTM model
 * the variability of text synthesis
 * the length of the synthesized text

Descriptions of the parameters are provided in [p3b1.py](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1.py).

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

To train the model from scratch, run [p3b1_baseline_keras2.py](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1_baseline_keras2.py). 

```
python p3b1_baseline_keras2.py
```

This script does the following:
 * Downloads and uncompresses the preprocessed data file from MoDaC.
 * Prepares the training and testing data.
 * Builds the multitask DNN model.
 * Trains the model.
 * Prints the performance metrics of the model for each task like the example below.

```
Task 1 : Primary site - Macro F1 score 0.15679785330948118
Task 1 : Primary site - Micro F1 score 0.3838383838383838
Task 2 : Tumor laterality - Macro F1 score 0.6312178387650085
Task 2 : Tumor laterality - Micro F1 score 0.6511627906976745
Task 3 : Histological grade - Macro F1 score 0.2923280423280423
Task 3 : Histological grade - Micro F1 score 0.4852941176470588
Average loss:  1.1945778330167134
```

## Acknowledgments
   
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.

**Overview**: Given a corpus of patient-level clinical reports, build a deep learning network that can simultaneously identify: (i) b tumor sites, (ii) t tumor laterality, and (iii) g clinical grade of tumors.

**Relationship to core problem**: Instead of training individual deep learning networks for individual machine learning tasks, build a multi-task DNN that can exploit task-relatedness to simultaneously learn multiple concepts.

**Expected outcome**: Multi-task DNN that trains on same corpus and can automatically classify across three related tasks.

### Benchmark Specs

#### Description of data
* Data source: Annotated pathology reports
* Input dimensions: 250,000-500,000 [characters], or 5,000-20,000 [bag of words], or 200-500 [bag of concepts]
* Output dimensions: (i) b tumor sites, (ii) t tumor laterality, and (iii) g clinical grade of tumors

* Sample size: O(1,000)
* Notes on data balance and other issues: standard NLP pre-processing is required, including (but not limited to) stemming words, keywords, cleaning text, stop words, etc. Data balance is an issue since the number of positive examples vs. control is skewed

#### Expected Outcomes
* Classification
* Output range or number of classes: Initially, 4 classes; can grow up to 32 classes, depending on number of tasks simultaneously trained.

#### Evaluation Metrics
* Accuracy or loss function: Standard approaches such as F1-score, accuracy, ROC-AUC, etc. will be used.
* Expected performance of a naïve method: Compare performance against (i) deep neural nets against single tasks, (ii) multi-task SVM based predictions, and (iii) random forest based methods.

#### Description of the Network
* Proposed network architecture: Deep neural net across individual tasks
* Number of layers: 5-6 layers

A graphical representation of the model is shown below:
![Multilayer DNN Architecture]<img src="https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/multilayer-dnn.png" width="200" height="200">

### Running the baseline implementation
There are two broad options for running our MTL implementation. The first baseline option includes the basic training of an MTL-based deep neural net. The second implementation includes a standard 10-fold cross-validation loop and depends on the first baseline for building and training the MTL-based deep neural net.

For the first baseline run, an example run is shown below:
```
cd P3B1
python MTL_run.py
```

For the second baseline run, including the 10-fold cross-validation loop, the set up is shown below.
```
cd P3B1
python keras_p3b1_baseline.py
```

Note that the training and testing data files are provided as standard CSV files in a folder called P3B1_data. The code is documented to provide enough information to reproduce the code on other platforms.






