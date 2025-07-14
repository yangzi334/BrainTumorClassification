# Brain tumor classification 

This repository contains data and code for JMIR paper:
# Trade-off Analysis of Classical and Deep Learning Models for Robust Brain Tumor Detection 

#Project Structure
- We first trained four models on the main dataset and evaluated their robustness using unseen test samples from the main dataset. All .ipynb python programs without the '_extension' corresponded to this main dataset.
- To assess the transferability and generalization, we retrained the same four models on the external dataset from a different source, using the same paramter settings, augmentation strategies, and training pipelines as those usdef for the main dataset.All .ipynb python programs with the '_extension' associated with the external dataset. 

# From the main datast: 
-preprocessing.ipynb: Code for data preprocessing.
This notebook checks COCO format annotation files under the training, validation, and test folders. It fixes mismatched annotation IDs in the training folder. Validation and test annotation files are correctly aligned with image IDs.

-ML_baseline_with_HOG_SVM.ipynb: Classical machine learning pipeline using SVM with HOG features.

-CNN_baseline_with_ResNet18.ipynb: Convolutional neural network (CNN) training using the ResNet18 architecture.

-Transformer_baseline_with_Vit.ipynb: Vision Transformer (ViT-B/16) deep learning model training and evaluation.

-Self_learning_baseline.ipynb: Self-supervised learning using SimCLR architecture.

-data/ folder: Contains train, valid, and test subfolders:

-train/: Includes 1,502 training images, the original annotation file _annotation.coco.json, and the corrected annotation file fixed_annotation.coco.json.

-valid/: Includes 429 validation images and _annotation.coco.json. All annotations are correctly matched with image IDs; no correction needed.

-test/: Includes 215 test images and _annotation.coco.json.


# From the external datast: 
-preprocessing_extension.ipynb: Code for data preprocessing.
This scripts prepares the external dataset by organizing the data into the same structure as the main dataset(train, valid, test folders).
     first: create the following folders:
  - ‘..\train\tumor’
  - ‘..\validation\tumor’
  - ‘..\test\tumor’
  - ‘..\train\no-tumor’
  - ‘..\validation\no-tumor’
  - ‘..\test\no-tumor’
    second: combined three tumor types - glioma, meiningioma, and pituitary into a single tumor class, then split the 2477 tumor images and 395 no-tumor images
    into train, validation, and test sets using the ratio of 0.7, 0.15, 0.15, saving them into the corresponding folders.  
  - we created the folders ‘..\train\tumor’, ‘..\validation\tumor’, and ‘..\test\tumor’, ‘..\train\no-tumor’, ‘..\validation\no-tumor’, and ‘..\test\no-tumor’ to  

-ML_baseline_with_HOG_SVM_extension.ipynb: Classical machine learning pipeline using SVM with HOG features.

-CNN_baseline_with_ResNet18_extension.ipynb: Convolutional neural network (CNN) training using the ResNet18 architecture.

-Transformer_baseline_with_Vit_extension.ipynb: Vision Transformer (ViT-B/16) deep learning model training and evaluation.

-Self_learning_baseline_extension.ipynb: Self-supervised learning using SimCLR architecture.


## models folder
# From the main datast: 
-base_SVM_HOG_new/: Stores the best-performing SVM+HOG model as best_svm_hog_model.pkl.

-base_resnet18_new/, base_Vit_B_16_new/, base_simclr_new/: 
Each contains models from individual training runs and the best model (lowest validation loss overall).
The SimCLR folder also includes the encoder parameter file.

-test_unseen_evaluation.ipynb: Evaluates the best-performing models (SVM+HOG, ResNet18, ViT-B/16, SimCLR) on unseen test data.

# From the external datast: 

-base_SVM_HOG_new_extension/: Stores the best-performing SVM+HOG model as best_svm_hog_model_extension.pkl.

-base_resnet18_new_extension/, base_Vit_B_16_new_extension/, base_simclr_new_extension/: 
Each contains models from individual training runs and the best model (lowest validation loss overall).
The SimCLR folder also includes the encoder parameter file.

-test_unseen_evaluation_extension.ipynb: Evaluates the best-performing models (SVM+HOG, ResNet18, ViT-B/16, SimCLR) on unseen test data.

## How to run
1: Install dependencies
Run: pip install -r requirements.txt
Alternatively, install libraries directly as they appear in the notebooks.

# First run programs associated to main dataset, including 'Data Proprocessing', 'trainig and evaluation models', 'test unseen program': 
2: Data preprocessing
Run preprocessing.ipynb to validate and clean COCO annotations for training data.

3: Train and evaluate baseline models

Run ML_baseline_with_HOG_SVM.ipynb for SVM+HOG.

Run CNN_baseline_with_ResNet18.ipynb for ResNet18.

Run Transformer_baseline_with_Vit.ipynb for ViT-B/16.

Run Self_learning_baseline.ipynb for SimCLR.

4: Compare results across all models 

Run final_evaluation_comparison_figures.ipynb to generate key comparison figures:

Figure: Average training vs. validation accuracy.

Figure: Training vs. validation loss.

Figure: Total training time across all models.

Note: The figures in final_evaluation_comparison_figures.ipynb (Figures 1–3) are generated using manually entered summary statistics (e.g., accuracy, loss, standard deviation, training time) collected from earlier model runs. These values are not computed dynamically in the notebook. Please refer to the corresponding training logs or evaluation outputs in each model folder for original values.To reproduce the plots: run final_evaluation_comparison_figures.ipynb after ensuring the pre-filled metrics in the notebook reflect the results from each model runs.

5: Test on unseen data
Run test_unseen_evaluation.ipynb to evaluate each model's performance on the test set.
Note: Before running this notebook, ensure that all trained models are available in their respective folders:

base_SVM_HOG_new/

base_resnet18_new/

base_Vit_B_16/

base_simclr_new/

# then run the external data associated programs, including 'Data Proprocessing', 'trainig and evaluation models', 'test unseen program': 

6: Data preprocessing
Run preprocessing_extension.ipynb 

7: Train and evaluate baseline models

Run ML_baseline_with_HOG_SVM_extension.ipynb for SVM+HOG.

Run CNN_baseline_with_ResNet18_extension.ipynb for ResNet18.

Run Transformer_baseline_with_Vit_extension.ipynb for ViT-B/16.

Run Self_learning_baseline_extension.ipynb for SimCLR.

8: Compare results across all models 

Run final_evaluation_comparison_figures_extension.ipynb to generate key comparison figures:

Figure: Average training vs. validation accuracy.

Figure: Training vs. validation loss.

Figure: Total training time across all models.

Note: The figures in final_evaluation_comparison_figures.ipynb (Figures 1–3) are generated using manually entered summary statistics (e.g., accuracy, loss, standard deviation, training time) collected from earlier model runs. These values are not computed dynamically in the notebook. Please refer to the corresponding training logs or evaluation outputs in each model folder for original values.To reproduce the plots: run final_evaluation_comparison_figures_extension.ipynb after ensuring the pre-filled metrics in the notebook reflect the results from each model runs.

9: Test on unseen data
Run test_unseen_evaluation_extension.ipynb to evaluate each model's performance on the test set.
Note: Before running this notebook, ensure that all trained models are available in their respective folders:

base_SVM_HOG_new_extension/

base_resnet18_new_extension/

base_Vit_B_16_extension/

base_simclr_new_extension/

## data availability
All fine-tuned model files and multiple appendixes and the raw brain tumor image dataset used in this study are openly available on Zenodo at the following DOI: https://zenodo.org/uploads/15258590
All code used in this study, along with a detailed README.md file explaining how to reproduce the results, is available on GitHub: 
https://github.com/yangzi334/BrainTumorClassification


