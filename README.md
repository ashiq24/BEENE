# BEENE
**BEENE** (Batch Effect Esitimation using Non-linear Embeddings) is a deep learning based technique for estimating batch effect in RNA-seq data. The variation introduced in the data by techniqal non-biological factors is called batch effect. It is very crutical to detect the extent of batch effect present in the data and remove it for unbiased analysis. Current batch effect estimatimation techniques depends on local distribution of cells in the PC space, which fail to camputre highly non-linear batch effect. To the best of our knowledge , BEENE is the first technique that provide non-linear embedding for batch effect estimation and is shwon to capture non-linear batch efect in RNA-seq data.

## Requirements
 Python >= 3.7.0
 
 For installing additional requirements from **requirements.txt** run the following command
 
 ```console
  pip install -r requirements.txt
 ```
 
## Usage
### calculating LISI metric with non-linear embedding

Lisi is a batch effect estimation metric [Link](https://github.com/immunogenomics/LISI). With the **BEENE** package LISI metric can be calculated directly using non-linear embedding. The class **beene_model** contains all necessery function implemented

The functions descriptions and parameters
```java
get_hybrid_model_1(number_of_genes, hidden_layer_dimensions, embedding_dimension,  number_of_batches,  number_of_biological_variables,  reconstruction_weight, batch_weight, bio_weight, islarge= False):
        """
        Creates a BEENE model
        
        Parameters:
        number_of_genes: int, Number of gene per sample in the data 
        hidden_layer_dimensions: list, hidden_layer_dimensions[0] is the dimension of 1st hidden layer hidden_layer_dimensions[1] is the dimension of 2nd hidden layer.    
        embedding_dimension: int, Dimension of the BEENE Embedding
        number_of_batches: int, Number of classes of the Batch variable
        number_of_biological_variables: int,  Number of classes of the Biological variable
        reconstruction_weight: float, weight for acutoencoder reconstruction loss
        batch_weight: float,  weight for batch label prediction error 
        bio_weight: float, weight of biological label prediction error
        islarge : bool, Set to true for high dimensional dataset. Model uses additional dropout in the input layer for high dimensional data.
        """
```

