# Brain tumor classification 

This repository contains data and code for JMIR paper:
# Trade-off Analysis of Classical and Deep Learning Models for Robust Brain Tumor Detection 

#Project Main Structure
- 1: Methods - Data Description, Data Preprocessing, Model designing, and Model Training.
- 2: Results - Training behavior and Convergence analysis, Train and Validation data performance, Generalization test on unseen data within and across domains, Visual interpretation via Saliency Map. 
- 3: Disuccions - Priniciple findings, Limitatons, etc. 

The project's details can be found in the paper.

## How to run
1: Install dependencies
Run: pip install -r requirements.txt
Alternatively, install libraries directly as they appear in the notebooks.

2: Run preprocessing.ipynb (Corresponding with the Data Preprocessing step based on the main dataset, the details of the Data processing are in the paper)

3: Run training model files, including: SVM_HOG.ipynb, ResNet18.ipynb, Vit_B_16.ipynb, and SimCLR.ipynb

4: Run evaluation_figure.ipynb, to show the figures comparison across four models, including 'training accuracy and validation accuracy', 'training loss and validation loss', 'training time' . 

5: Run test_unseen_within_domain.ipynb, to show the classification table, confusion matrices, and saliency maps for each of model on the unseen test data within the main domain.

Note: from step 1 to step 5, we done the training and validation and testing based on the main dataset, from the next step, we start to work on external image data. 

6: Run 'check image leakage.ipynb' to use phash algorithm to compare each external image with training images, any viusally identical or nearly identical images were removed from the external dataset. 

7: Run preprocessing_external.ipynb, to preprocess the external image data, such as grouping them into 'tumor' and 'non-tumor', to maintain consistency with the structure of the main dataset. 

8: Run test_unseen_across_domains.ipynb, to evaluate all models on the external datasets without retraining to better reflect real-world deployment.


## data availability
All fine-tuned model files and multiple appendixes and the raw brain tumor image datasets used in this study are openly available on Zenodo at the following DOI: [https://zenodo.org/uploads/16749990]
All code used in this study, along with a detailed README.md file explaining how to reproduce the results, is available on GitHub: 
https://github.com/yangzi334/BrainTumorClassification


