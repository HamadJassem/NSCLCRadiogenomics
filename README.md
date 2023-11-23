# Multi-modal Medical Image Fusion to Segment and Classify Non-Small Cell Lung Cancer

In this project, we address the challenging problem of early detection and accurate classification of non-small cell lung cancer (NSCLC), a leading cause of cancer-related deaths globally. Our approach integrates multi-modal data, combining  fused medical imaging (CT and PET scans) with clinical health records and genomic data. Utilizing a novel fusion of these diverse data sources, we employ advanced machine learning models, including MedClip and BEiT for image feature extraction, baseline models for comparisons and a UNet model for precise tumor segmentation. Key findings demonstrate a significant improvement in the accuracy and precision of NSCLC detection and classification by using the fused data, as evidenced by enhanced performance metrics such as accuracy, precision, recall, and F1-score. The best multi-modal classifier model achieved an accuracy of 94.04\% and the segmentation model achieved a DSC of 0.76. This project's significance lies in its potential to revolutionize NSCLC diagnostics, offering earlier detection, improved treatment planning, and contributing to better patient outcomes in the realm of lung cancer treatment. 

<p align="center">
  <img src="https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/imgs/model_arch.png" alt="Model Architecture Diagram"/>
</p>
<p align="center">
  <img src="https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/imgs/seg.png" alt="Model Architecture Diagram"/>
</p>

## Fusion model output
<p align="center">
  <img src="https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/imgs/output.png" alt="Fusion Model Output"/>
</p>

## Segmenattion model output
<p align="center">
  <img src="https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/imgs/segmentation.png" alt="Segmentation Output"/>
</p>

---
### Directory Guide:
- #### src Notebooks
  - [00_Downloading_Data_From_TCIA](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/00_Downloading_Data_From_TCIA.ipynb) -> used to get the dataset (CT, PET) from the cancer imaging archive
  - [01_Deep_CNN_Autoencoder_Denoising_Image](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/01_Deep_CNN_%20Autoencoder_Denoising_Image.ipynb) -> Deep CNN autoencoder model to denoise and pre-process the PET scans
  - [02_CT_Lung_Mask_Prosessing](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/02_CT_Lung_Mask_Prosessing.ipynb) -> CT scans pre-processing and filtering
  - [03_Transfer_Learning - Fusion](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/03_Transfer_Learning%20-%20Fusion.ipynb) -> VGG19 Fusion model to fuse the CT and PET image slices
  - [04_Genetic_Data_Pre-Processing](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/04_Genetic_Data_Pre-Processing.ipynb) -> Genetic data pre-processing and fetaure section
  - [05_Clinical_Data_Pre-Processing](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/05_Clinical_Data_Pre-Processing.ipynb) -> Clinical data cleaning and pre-processing and feature selction
  - [06_Pyradiomics_CT_Feature_Extraction](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/06_Pyradiomics_CT_Feature%20Extraction.ipynb) -> CT image feature extraction using pyradiomics and tumor mask
  - [07_Image_Processing_resampling_Fusion_Slicing_Pipeline](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/07_Image_Processing%20_resampling_Fusion_Slicing_Pipeline.ipynb) -> Pipeline to perform image pre-processing, fusion and slicing to filter relevant Lung images
  - [08_Lung_Tumor_Segmentation](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/08_Lung_Tumor_Segmentation.ipynb) -> UNet model to segment the NSCLC tumor
  - [09_SVM_Model](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/09_SVM_Model.ipynb) -> SVM model for clinical and genetic data only with GridSearch
  - [10_Logistic_Regression_Model](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/10_Logistic_Regression_Model.ipynb) -> Logistic Regression model for clinical and genetic data only with GridSearch
  - [11_2D_CNN_Image_Classifier](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/11_2D%20CNN%20Image%20Classifier.ipynb) -> 2D CNN model for image classification - change directory to switch between CT and Fused CT/PET images
  - [12_3D_CNN_With_DenseNet](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/12_3D_CNN_With_DenseNet.ipynb) -> 3D CNN DenseNet architecture for nrrd file classification - change directory to switch between CT and Fused CT/PET images
  - [13_Transfer_Learning_ResNet_Fused](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/13_Transfer_Learning_ResNet_Fused.ipynb) -> ResNet model for image classification on Fused images
  - [14_Transfer_Learning_ResNet_CT](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/14_Transfer_Learning_ResNet_CT.ipynb) -> ResNet model for image classification on CT images
  - 15 VGG16
  - 16 Inception
  - 17 Xception
  - [18_Multimodal_MedCLIP_CT](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/18_Multimodal_MedCLIP_CT.ipynb) -> Multi-modal MedClip based model for classification based on CT images, clinical data and genomics data
  - [19_Multimodal_MedCLIP_CT_PET](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/19_Multimodal_MedCLIP_CT_PET.ipynb) -> Multi-modal MedClip based model for classification based on CT images, PET images, clinical data and genomics data
  - [20_Multimodal_MedCLIP_Fused](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/20_Multimodal_MedCLIP_Fused.ipynb) -> Multi-modal MedClip based model for classification based on Fused CT/PET images, clinical data and genomics data
  - [21_Multimodal_BEiT_CT](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/21_Multimodal_BEiT_CT.ipynb) -> Multi-modal BEiT based model for classification based on CT images, clinical data and genomics data
  - [22_Multimodal_BEiT_CT_PT](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/22_Multimodal_BEiT_CT_PT.ipynb) -> Multi-modal BEit based model for classification based on CT images, PET images, clinical data and genomics data
  - [23_Multimodal_BEiT_Fused](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/src%20notebooks/23_Multimodal_BEiT_Fused.ipynb) -> Multi-modal BEiT based model for classification based on Fused CT/PET images, clinical data and genomics data

---

- #### Data
  - [Fused Lung](https://github.com/HamadJassem/NSCLCRadiogenomics/tree/main/Data/Fused%20Lung%202%20copy) -> fused CT and PET images save as: PatientID_Slice#_Fused.jpg
  - [Dataset Lung img](https://github.com/HamadJassem/NSCLCRadiogenomics/tree/main/Data/Dataset%20Lung%20img) -> contains subfolder for each patient. Each folder will have CT, PET slices as well as the denoised PET scan. CT image saved as: PatientID_Slice#_CT.jpg, PT scan: PatientID_Slice#_PT.jpg, Denoised PET: PatientID_Slice#_PT_denoised.jpg
  - [Lung Mask](https://github.com/HamadJassem/NSCLCRadiogenomics/tree/main/Data/Lung%20Mask%20Train%20Test) -> contains train and test subfolders. train and test have 2 subfolders each for class 0 and class 1 each with the CT scans after applying a lung mask. File name: PatientID_CT_Slice#.jpg
  - [Full CT Lung](https://github.com/HamadJassem/NSCLCRadiogenomics/tree/main/Data/Full%20CT%20Lung) -> contains full CT lung images, File name: PatientID_CT_Slice#.jpg
  - [GSE103584_R01_NSCLC_RNAseq.txt](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/Data/GSE103584_R01_NSCLC_RNAseq.txt) -> raw genomics data of the patient subset
  - [clinical_subset.csv](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/Data/clinical_subset.csv) -> raw clinical data of the patient subset
  - [train_data_resamples_reordered.csv](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/Data/train_data_resamples_reordered.csv) -> pre-processed and cleaned clinicla and genetic features for training after SMOTE class balancing and feature selection
  - [test_data_resamples_reordered.csv](https://github.com/HamadJassem/NSCLCRadiogenomics/blob/main/Data/test_data_resamples_reordered.csv) -> pre-processed and cleaned clinicla and genetic features for testing after SMOTE class balancing and feature selection
---
