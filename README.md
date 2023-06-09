# BEENE
**BEENE** (Batch Effect Estimation using Non-linear Embeddings) is a deep learning-based technique for estimating batch effects in RNA-seq data. The variations introduced to the data by technical non-biological factors are known as batch effects. It is critical to detect the extent of batch effects present in the data and remove it for unbiased downstream analysis. Currently, the batch effect estimation techniques mostly depend on the local distribution of cells in the Principal Component (PC) space, which may fail to capture sophisticated non-linear batch effects. BEENE is the first known method to explicitly consider and model the non-linearity of batch effects for effective batch effect detection and estimation. BEENE simultaneously learns the batch and biological variables from RNA-seq data, resulting in an embedding that is more robust and sensitive than PCA embedding in terms of detecting and quantifying batch effects.

## Requirements
 Python >= 3.7.0
 
 For installing additional requirements from **requirements.txt** run the following command
 
 ```console
  pip install -r requirements.txt
 ```
 
## Usage
### Description and Parameters of Major Functions
The class _beene_model_ has all the necessary functions for building the BEENE model, training the model, getting the embeddings, and as well as calculating iLISI values. Some member functions of this class are:
```PYTHON
get_hybrid_model_1(number_of_genes, hidden_layer_dimensions, embedding_dimension,  number_of_batches,\
number_of_biological_variables,  reconstruction_weight, batch_weight, bio_weight, islarge= False):
        """
        Creates and returns the BEENE model.
       
        
        Parameters:
        number_of_genes: int, Number of genes per sample in the data 
        hidden_layer_dimensions: list, hidden_layer_dimensions[0] is the dimension of 1st hidden layer and 
                                 hidden_layer_dimensions[1] is the dimension of 2nd hidden layer.
       
        embedding_dimension: int, Dimension of the BEENE Embedding
        number_of_batches: int, Number of classes of the Batch variable
        number_of_biological_variables: int,  Number of classes of the Biological variable
        reconstruction_weight: float, weight for auto-encoder reconstruction loss
        batch_weight: float,  weight for batch label prediction error 
        bio_weight: float, weight of biological label prediction error
        islarge : bool, Set to true for high dimensional dataset. The model uses a higher dropout rate in the 
                 input layer for high dimensional data.
        """
 evaluate_batch_iLisi(data, batch_var, bio_var, seed):

      """
      Compute the iLISI metric on the data. The data is split into test, training, and validation
      set. The model defined by the user is trained by the training set. The best-performing model
      is selected by the validation set. And the iLISI index is calculated on the test set.
      
      returns: iLISI values of each of the samples in the randomly chosen test set using BEENE embedding
      
      Parameters:
      data: 2D numpy array. Each row represents a sample and each column represents a Gene.
      batch_var: numpy array. For more than two categories, it must be one hot representation of 
                batch labels for each of the samples in the data and must be a dense matrix. For 
                two categories, it must be a 1D array of zeros and ones denoting batch association 
                for each of the samples in the data
      bio_var: numpy array. For more than two categories, it must be one hot representation of batch 
              labels for each of the samples in the data and must be a dense matrix. For two categories,
              it must be a 1D array of zeros and ones denoting the biological class for each of the samples in the data
      seed: int. Random state for the split
      """
 train_model(train_x, train_batch,train_bio,val_x,val_batch,val_bio, num_epochs, batch_size = 32):
    """
    Training the defined BEENE model
    
    Parameters:
    train_x: np.ndarray, Training set: Gene expressing 
    train_batch: np.ndarray, Training set: Corresponding batch variables 
    train_bio: np.ndarray, Training set: Corresponding Biological variables (None if biological variables are absent)
    val_x: np.ndarray, Validation set: Gene expressing 
    val_batch: np.ndarray, Validation set: Corresponding batch variables
    val_bio: np.ndarray, Validation set: Corresponding Biological variables (None if biological variables are absent)
    num_epochs: int, Number of epochs to train the model
    batch_size : int, batch size (default 32)
    
    """
 
 get_beene_embeddings(data):
      """
      returns the embedding learn by the model for the **data**
      
      Parameters
      data: np.ndarray
      """
 
```
### Choice of Hyperparameters
*reconstruction_weight* ($\lambda_1$), *batch_weight* ($\lambda_2$), and *bio_weight* ($\lambda_3$) are three important parameters of BEENE. When *batch_weight* is set comparatively higher than *reconstruction_weight* and  *bio_weight*, the cell embeddings will be more tightly clustered by thier respective batches if batch effect is present in the data. In other words, BEENE will be more sensetive to the batch effect if the *batch_weight* parameter is set higher.

The choice of these hyperparameters is dependent on the data and the downstream application. We recommend to use  $\lambda_1 = \lambda_3 = 1$ and $\lambda_2 = 2$ when a dimensionality reduction technique has been used to reduce the number of features to hundreds. When the dataset is used without applying any dimensionality reduction technique with the number of features in the range of thousands, the value of $\lambda_1$ should be decreased accordingly. (eg 1/1000).

| Biological Feature | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ | Dimensionality reduction applied?                                 |
|--------------------|:-----------:|:-----------:|:-----------:|-------------------------------------------------------------------|
| Present            |    1.000    |     2.0     |     1.0     | Yes (i.e., the number features is in the range of a few hundreds) |
|                    |    0.001    |     5.0     |     5.0     | No (i.e., the number features is in the range of thousands)       |
| Absent             |    1.000    |     1.0     |      -      | Yes                                                               |
|                    |    0.001    |     5.0     |      -      | No                                                                |

### Generating and Storing BEENE-Embeddings
Our tool BEENE creates low dimensional embeddings (BEENE-Embeddings) from RNAseq data which are very effective for batch effect assessment. An example for generating and storing BEENE-Embeddings from RNA-seq data is shown below

Importing necessary packages

```python
from beene import beene_model
from numpy import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
```

creating simulated  data with 3000 samples and 100 genes per sample, with 2 biological categories and 3 batches.
```python
Xt = random.uniform(-1,1,(3000,100))
# biological variables
yt = random.randint(0,2,3000)
# batches 
bt = random.randint(0,3,3000)
````
Creating the BEENE model
with embedding dimension of 5
Encoder Network: The size of the first hidden layer is 50, and the second hidden layer is 20
The Encoder and the Decoder networks are symmetric.
reconstruction loss weight: 1
batch prediction loss weight: 2
biological variables prediction loss weight: 2

```python
my_model = beene_model()
my_model.get_hybrid_model_1(100,[50,20],5,3,2,1,2,1)
```

Creating one-hot vectors for batch variables. As the number of classes in biological variables is 2, creating one-hot vectors is not necessary

```python
bt = np.reshape(bt,(-1,1))
enc_bi = OneHotEncoder(handle_unknown='ignore')
enc_bi.fit(bt)
bt = enc_bi.transform(bt)
bt = bt.todense()
```


Creating training-validation-test split
```python
X_train, X_test, Y_Platform_train, Y_Platform_test,Y_ER_train,Y_ER_test = train_test_split(
                                          Xt, bt, yt,test_size=0.20,random_state=4)

#Getting separate validation data
X_train, X_val, Y_Platform_train, Y_Platform_val,Y_ER_train,Y_ER_val = train_test_split(
                                           X_train, Y_Platform_train, Y_ER_train ,test_size=0.25,random_state=4)
```
Training the model for 300 epochs
```python
my_model.train_model(X_train,Y_Platform_train,Y_ER_train,X_val,Y_Platform_val,Y_ER_val,100)
```

Saving embedding for the test set. Embeddings will be saved in the 'txt' format. Embeddings for each of the cells will be stored along
the rows. Values are space-separated. 

```python
test_embedding = my_model.get_beene_embeddings(X_test)
np.savetxt('embedding.txt', test_embedding, fmt='%f')
# for loading 
loaded_embedding = np.loadtxt('embedding.txt', dtype=float)
```
### Calculating LISI Metric with Non-linear Embedding

Lisi is a batch effect estimation metric [Link](https://github.com/immunogenomics/LISI). With the **BEENE** package, LISI metric can be calculated using non-linear embedding directly from the data. The class **beene_model** contains all necessary functions implemented


An example is provided here for calculating LISI metrics from data [example_1.py](https://github.com/ashiq24/BEENE/blob/main/beene/example_1.py). It is explained below.

Importing the **beene** package and other required modules for data loading, generation, and preprocessing. The **beene.py** should be in the same folder as your python script.
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

Creating the one-hot vector for the batch variables.

```python
bt = np.reshape(bt,(-1,1))
enc_bi = OneHotEncoder(handle_unknown='ignore')
enc_bi.fit(bt)
bt = enc_bi.transform(bt)
bt = bt.todense()

# Number of classes in biological variables is 2. 
# So creating one-hot vector is not necessary

```

Calculating LISI using nonlinear embedding.

```python
# calculating iLisi values for the data
lisi_values = my_model.evaluate_batch_iLisi(Xt,bt,yt,20)

print(np.median(lisi_values))
```


### Calculating the kBET Metric with Non-linear Embedding

### Additional Requirements
R >= 4.1.0

rpy2 >= 3.4.0 (For seamless working this package Linux system is recommended)

kBET (Follow the instructions given [here](https://github.com/theislab/kBET/) for installation)


The [kBET](https://github.com/theislab/kBET/) does not have any python implementation available and is available only in R. To calculate kBET using non-linear embedding, the learned embedding must be transferred to R environment for calculation. 
An example of calculating kBET using non-linear embedding is given here [example_2.ipynb](https://github.com/ashiq24/BEENE/blob/main/beene/example_2.ipynb). It uses the **rpy2** package, which needs to be installed separately using the following command.
```console
pip install rpy2
```
The documentation for **rpy2** can be found [here](https://rpy2.github.io/doc/latest/html/index.html)

## Bug Report
ashiqbuet14@gmail.com

