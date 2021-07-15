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

Lisi is a batch effect estimation metric [Link](https://github.com/immunogenomics/LISI). With the **BEENE** package LISI metric can be calculated using non-linear embedding directly from the data. The class **beene_model** contains all necessery function implemented

#### Description and parameters of major functions
```java
get_hybrid_model_1(number_of_genes, hidden_layer_dimensions, embedding_dimension,  number_of_batches,  number_of_biological_variables,  reconstruction_weight, batch_weight, bio_weight, islarge= False):
        """
        Creates and return the BEENE model.
       
        
        Parameters:
        number_of_genes: int, Number of gene per sample in the data 
        hidden_layer_dimensions: list, hidden_layer_dimensions[0] is the dimension of 1st hidden layer and 
                                 hidden_layer_dimensions[1] is the dimension of 2nd hidden layer.
       
        embedding_dimension: int, Dimension of the BEENE Embedding
        number_of_batches: int, Number of classes of the Batch variable
        number_of_biological_variables: int,  Number of classes of the Biological variable
        reconstruction_weight: float, weight for acutoencoder reconstruction loss
        batch_weight: float,  weight for batch label prediction error 
        bio_weight: float, weight of biological label prediction error
        islarge : bool, Set to true for high dimensional dataset. Model uses additional dropout in the 
                 input layer for high dimensional data.
        """
 evaluate_batch_iLisi(data, batch_var, bio_var, seed):

      """
      Compute the iLISI metric on the data. The data is split into testm training, and validation
      set. The model defined by the user is trained by the training set. The best performing model
      is selected by the validation set. And the iLISI index is calculated on the test set.
      
      returns: iLISI values of each of the samples in the randomly choosen test set using BEENE embedding
      
      Parameters:
      data: 2D numpy array. Each row represents a sample and each column represents a Gene
      batch_var: numpy array. For more than two categories, it must be one hot representation of 
                batch labels for each of the samples in the data and must be a dense matrix. For 
                two categories, it must be a 1D array of zeros and ones denoting batch association 
                for each samples in the data
      bio_var: numpy array. For more than two categories, it must be one hot representation of batch 
              labels for each of the samples in the data and must be a dense matrix. For two categories,
              it must be a 1D array of zeros and ones denoting the biological class for each samples in the data
      seed: int. Random state for the split
      """
      
 get_beene_embeddings(data):
      """
      retuns the embedding learn by the model for the **data**
      
      Parameters
      data: np.ndarray
      """
 
```
An example if provided here for calculating LISI metric from data. [example_1.py](https://github.com/ashiq24/BEENE/blob/main/beene/example_1.py)


### calculating kBET metric with non-linear embedding

The [kBET](https://github.com/theislab/kBET/) does not have any python implementation available and available only in R. For calculate kBET using non-linear embedding, the learned embedding must be transferred to R enviroment for calculation. 

