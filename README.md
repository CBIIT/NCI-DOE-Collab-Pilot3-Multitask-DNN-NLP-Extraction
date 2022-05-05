# NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction

### Description
Given a sample corpus of biomedical text (i.e. pathology reports), this resource builds a multitask deep neural network (DNN) to automatically extract different types of information (i.e. tumor sites and tumor laterality).

### User Community
Data scientists interested in information extraction from unstructured texts such as clinical pathology reports.

### Usability	
Data scientists can train the provided untrained model on their own data or with preprocessed data of clinical pathology reports included with this resource. 

To use this resource, users must be familiar with natural language processing (NLP) and training neural networks.

### Uniqueness	
Instead of training individual DNNs for single tasks, this resource leverages the relatedness of several tasks to build a DNN capable of multiple tasks. 

### Components	
* Script to train a multitask DNN model: [p3b1_baseline_keras2.py](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-DNN-NLP-Extraction/blob/master/Pilot3/P3B1/p3b1_baseline_keras2.py)
* Data: The preprocessed training and test data of clinical pathology reports are in [MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-18269439).

### Technical Details
Refer to this [README](./Pilot3/P3B1/README.md).

### Reference

For more details, refer to this [publication](https://link.springer.com/chapter/10.1007/978-3-319-47898-2_21).
