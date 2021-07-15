# BEENE
**BEENE** (Batch Effect Esitimation using Non-linear Embeddings) is a deep learning based technique for estimating batch effect in RNA-seq data. The variation introduced in the data by techniqal non-biological factors is called batch effect. It is very crutical to detect the extent of batch effect present in the data and remove it for unbiased analysis. Current batch effect estimatimation techniques depends on local distribution of cells in the PC space, which fail to camputre highly non-linear batch effect. To the best of our knowledge , BEENE is the first technique that provide non-linear embedding for batch effect estimation and is shwon to capture non-linear batch efect in RNA-seq data.

## Requirements
 Python >= 3.7.0
 
 For installing additional requirements from **requirements.txt** run the following command
 
 ```console
  pip install -r requirements.txt
 ```

