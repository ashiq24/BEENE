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
        reconstruction_weight: float, weight for acut-encoder reconstruction loss
        batch_weight: float,  weight for batch label prediction error 
        bio_weight: float, weight of biological label prediction error
        islarge : bool, Set to true for high dimensional dataset. Model uses additional dropout in the 
                 input layer for high dimensional data.
        """
 evaluate_batch_iLisi(data, batch_var, bio_var, seed):

      """
      Compute the iLISI metric on the data. The data is split into test, training, and validation
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
An example if provided here for calculating LISI metric from data [example_1.py](https://github.com/ashiq24/BEENE/blob/main/beene/example_1.py). It is explained below.

Importing beene package and other required modules for data loading, generation and preprocessing. The **beene.py** should in the same folder as your python script.
```python
from beene import beene_model
from numpy import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
```
Generating example data

```python
# creating Random data with 3000 samples and 100 genes per sample.
Xt = random.uniform(-1,1,(3000,100))
# With 2 biological categories 
yt = random.randint(0,2,3000)
# With 3 batches
bt = random.randint(0,3,3000)
```
Creating the BEENE model

```python
# Creating the BEENE model
# with embedding dimension of 5
# Size of first hiddent layer is 50, and second hidden
# layer is 20
# reconstruction_weight: 1,
# batch_weight: 2,
# bio_weight: 2,

my_model = beene_model()
my_model.get_hybrid_model_1(100,[50,20],5,3,2,1,2,1)

```

Creating the onehot vector for the batch variables.

```python
bt = np.reshape(bt,(-1,1))
enc_bi = OneHotEncoder(handle_unknown='ignore')
enc_bi.fit(bt)
bt = enc_bi.transform(bt)
bt = bt.todense()

# Number of classes in biological variables is 2. 
# So creating one-hot vector is not necessery

```

Calculating LISI using nonlinear embedding.

```python
# calculating iLisi values for the data
lisi_values = my_model.evaluate_batch_iLisi(Xt,bt,yt,20)

print(np.median(lisi_values))
```


### calculating kBET metric with non-linear embedding

### Additonal Requirements
R >= 4.1.0

rpy2 >= 3.4.0 (For seamless working this package Linax system is recommended)


The [kBET](https://github.com/theislab/kBET/) does not have any python implementation available and available only in R. For calculate kBET using non-linear embedding, the learned embedding must be transferred to R enviroment for calculation. 
And example of calculating kBET using non linear embedding is give here [example_2.ipynb](https://github.com/ashiq24/BEENE/blob/main/beene/example_2.ipynb). It uses the **rpy2** package needs to be installed separately using the following command.
```console
pip install rpy2
```
The documentation for rpy2 can be found [here](https://rpy2.github.io/doc/latest/html/index.html)

## Bug Report
ashiqbuet14@gmail.com

