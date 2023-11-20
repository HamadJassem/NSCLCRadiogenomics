# NSCLCRadiogenomics

In this project, we address the challenging problem of early detection and accurate classification of non-small cell lung cancer (NSCLC), a leading cause of cancer-related deaths globally. Our approach integrates multi-modal data, combining  fused medical imaging (CT and PET scans) with clinical health records and genomic data. Utilizing a novel fusion of these diverse data sources, we employ advanced machine learning models, including MedClip and BEiT for image feature extraction, baseline models for comparisons and a UNet model for precise tumor segmentation. Key findings demonstrate a significant improvement in the accuracy and precision of NSCLC detection and classification by using the fused data, as evidenced by enhanced performance metrics such as accuracy, precision, recall, and F1-score. The best multi-modal classifier model achieved an accuracy of 93.9\% and the segmentation model achieved a DSC of 0.76. This project's significance lies in its potential to revolutionize NSCLC diagnostics, offering earlier detection, improved treatment planning, and contributing to better patient outcomes in the realm of lung cancer treatment. 

Directory Guide:
- src Notebooks
  - 00_Downloading_Data_From_TCIA -> used to get the dataset (CT, PET) from the cancer imaging archive
  - 01_Deep_CNN_Autoencoder_Denoising_Image -> Deep CNN autoencoder model to denoise and pre-process the PET scans
  - 02_CT_Lung_Mask_Prosessing -> CT scans pre-processing and filtering
  - 03_Transfer_Learning - Fusion -> VGG19 Fusion model to fuse the CT and PET image slices
  - 04_Genetic_Data_Pre-Processing -> Genetic data pre-processing and fetaure section
  - 05_Clinical_Data_Pre-Processing -> Clinical data cleaning and pre-processing and feature selction
  - 06_Pyradiomics_CT_Feature_Extraction -> CT image feature extraction using pyradiomics and tumor mask
  - 07_Image_Processing_resampling_Fusion_Slicing_Pipeline -> Pipeline to perform image pre-processing, fusion and slicing to filter relevant Lung images
  - 08_Lung_Tumor_Segmentation -> UNet model to segment the NSCLC tumor
  - 09_SVM_Model -> SVM model for clinical and genetic data only with GridSearch
  - 10_Logistic_Regression_Model -> Logistic Regression model for clinical and genetic data only with GridSearch
  - 11_2D_CNN_Image_Classifier -> 2D CNN model for image classification - change directory to switch between CT and Fused CT/PET images
  - 12_Transfer_Learning_ResNet_Fused -> ResNet model for image classification on Fused images
  - 13_Transfer_Learning_ResNet_CT -> ResNet model for image classification on Fused images
