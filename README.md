# Brain tumor classification 

This repository contains data and code for the JMIR paper:
# Trade-off Analysis of Classical and Deep Learning Models for Robust Brain Tumor Detection 

#Project Main Structure
- 1: Methods - Data Description, Data Preprocessing, Model designing, and Model Training.
- 2: Results - Training behavior and Convergence analysis, Train and Validation data performance, Generalization on unseen data (within-domain and cross-domain), and Visual Interpretation via Saliency Maps. 
- 3: Disuccions - Priniciple findings, Limitatons, Future Work, etc. 

For full project details, please refer to the paper.

## How to run
1: Install Dependencies
Run: pip install -r requirements.txt
Alternatively, install libraries directly as they appear in the notebooks.

2: Preprocessing the Primary Dataset
Run preprocessing.ipynb to preprocess the primary dataset(T1-weighted MRI)
Detailed data preprocessing steps are described in the paper. 

3: Train Models
Run the following notebooks to train each model: 
SVM_HOG.ipynb, ResNet18.ipynb, Vit_B_16.ipynb, and SimCLR.ipynb

4: Generate Evaluation Figures 
Run evaluation_figure.ipynb to produce comparison figures for the four models, including:
'training accuracy and validation accuracy', 'training loss and validation loss', 'training time' . 

5: Test on Unseen Primary-domain data
Run test_unseen_within_domain.ipynb to generate:
Classification tables, Confusion matrices, Saliency maps 
for all models using the unseen test data within the primary dataset.

Note: From step 1 to 5 focus on the primary dataset. From step 6 onward, the workflows uses the cross-dataset for external evaluation. 

6: Check for Image Leakage
Run 'check image leakage.ipynb' to use the perceptual hash(phash) algorithm to compare each cross-dataset image with training images. 
Any viusally identical or nearly identical images were removed from the cross-dataset. 

7: Preprocessing the Cross-Dataset
Run preprocessing_external.ipynb to preprocess the cross-dataset image, grouping images into 'tumor' and 'non-tumor', categories to maintain consistency with the structure of the primary dataset. 

8: Evaluate Cross-Domain Generalization
Run test_unseen_across_domains.ipynb to evaluate all models on the cross-dataset without retraining, simulating real-world deployment conditions.


## Data availability
All fine-tuned model files and multiple appendixes and the raw brain tumor image datasets used in this study are openly available on Zenodo at the following DOI: [https://zenodo.org/uploads/16749990]
All code used in this study, along with a detailed README.md file explaining how to reproduce the results, is available on GitHub: 
https://github.com/yangzi334/BrainTumorClassification


