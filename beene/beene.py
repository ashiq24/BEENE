import tensorflow as tf
from sklearn import datasets
from sklearn import preprocessing
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from numpy import random
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import time
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Iterable


def compute_lisi(
    X: np.array,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float=30
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.
    LISI is a statistic computed for each item (row) in the data matrix X.
    The following example may help to interpret the LISI values.
    Suppose one of the columns in metadata is a categorical variable with 3 categories.
        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.
        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.
    
    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].
    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors = perplexity * 3, algorithm = 'kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:,1:]
    distances = distances[:,1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:,i] = 1 / simpson
    return lisi_df


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float=1e-5
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:,i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:,i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:,i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson

class beene_model:
    """
    been_model class creats model for getting beene embeddings
    """


    def __init__(self,):
        print("Creating model")
        self.hybrid_model = None
        self.encoder_model = None
        self.model_checkpoint_callback = None
        
    
    def get_hybrid_model_1(self,
        number_of_genes: int, 
        hidden_layer_dimensions: list,
        embedding_dimension: int, 
        number_of_batches: int, 
        number_of_biological_variables: int, 
        reconstruction_weight: float,
        batch_weight: float,
        bio_weight: float,
        islarge : bool = False
        ):

        """
        Creates BEENE model

        Parameters:

        number_of_genes: int, Number of gene per sample in the data hidden_layer_dimensions: list, hidden_layer_dimensions[0] is the dimension of 1st hidden layer hidden_layer_dimensions[1] is the dimension of 2nd hidden layer.    
        embedding_dimension: int, Dimension of the BEENE Embedding
        number_of_batches: int, Number of classes of the Batch variable
        number_of_biological_variables: int,  Number of classes of the Biological variable
        reconstruction_weight: float, weight for acutoencoder reconstruction loss
        batch_weight: float,  weight for batch label prediction error 
        bio_weight: float, weight of biological label prediction error
        islarge : bool, Set to true for high dimensional dataset. Model uses additional dropout in the input layer for high dimensional data.

        """

        inputs=tf.keras.Input(shape=(number_of_genes,))

        #for high dimentional input
        inputs_1 = tf.keras.layers.Lambda(lambda x: 1*x )(inputs)
        if islarge:
            inputs_1 = tf.keras.layers.Dropout(0.70)(inputs_1)

        en_h1=tf.keras.layers.Dense(hidden_layer_dimensions[0],activation='selu',name='en_h1', kernel_regularizer='l2')(inputs_1)
        en_h1 = tf.keras.layers.Dropout(0.25)(en_h1)
        en_h2=tf.keras.layers.Dense(hidden_layer_dimensions[1],activation='selu',name='en_h2' , kernel_regularizer='l2')(en_h1)
        en_h2 = tf.keras.layers.Dropout(0.15)(en_h2)
        embedding=tf.keras.layers.Dense(embedding_dimension,activation='selu',name='embedding')(en_h2)

        encoder_model=tf.keras.Model(inputs=inputs,outputs=[embedding],name='encoder_model')

        #start of decoder
        de_h1=tf.keras.layers.Dense(hidden_layer_dimensions[1],activation='selu',name='de_h1', kernel_regularizer='l2')(embedding)
        de_h1 = tf.keras.layers.Dropout(0.15)(de_h1)
        de_h2=tf.keras.layers.Dense(hidden_layer_dimensions[0],activation='selu',name='de_h2', kernel_regularizer='l2')(de_h1)
        de_h2 = tf.keras.layers.Dropout(0.25)(de_h2)
        de_output=tf.keras.layers.Dense(number_of_genes,activation='selu',name='de_output', kernel_regularizer='l2')(de_h2)

        #start of platform detection layer
        if number_of_batches ==2:
            batch_output = tf.keras.layers.Dense(number_of_batches-1,activation='sigmoid',name='batch_output',  kernel_regularizer='l2')(embedding)
            batch_loss = "binary_crossentropy"
        else:
            batch_loss = "categorical_crossentropy"
            batch_output = tf.keras.layers.Dense(number_of_batches,activation='softmax',name='batch_output',  kernel_regularizer='l2')(embedding)

        if number_of_biological_variables == 0:
          bio_weight = 0

          #place holder for model
          number_of_biological_variables = number_of_batches

        #biological label detection
        if number_of_biological_variables ==2:
            bio_loss = "binary_crossentropy"
            bio_output = tf.keras.layers.Dense(number_of_biological_variables-1,activation='sigmoid',name='bio_output',  kernel_regularizer='l2')(embedding)
        else:
            bio_loss = "categorical_crossentropy"
            bio_output = tf.keras.layers.Dense(number_of_biological_variables,activation='softmax',name='bio_output',  kernel_regularizer='l2')(embedding)

        
        hybrid_model=tf.keras.Model(inputs=inputs,outputs=[batch_output,de_output,bio_output],name="hybridmodel")
        
        self.hybrid_model = hybrid_model
        self.encoder_model = encoder_model

        

        losses = {
                "batch_output": batch_loss,
                "de_output": tf.keras.losses.MeanSquaredError(),
                "bio_output": bio_loss,
                }
        lossWeights = {"batch_output":batch_weight, "de_output": reconstruction_weight,"bio_output":bio_weight}

        #model check pointer 
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose=2, save_best_only=True)
        self.hybrid_model.compile(optimizer=Adam(learning_rate=0.0005),
                                                    loss=losses,loss_weights=lossWeights,
                                            metrics={'batch_output': 'accuracy', 'bio_output': 'accuracy','de_output': 'mse'})
        
        return self.hybrid_model, self.encoder_model

    def train_model(self, 
                    train_x: np.ndarray, 
                    train_batch: np.ndarray, 
                    train_bio: np.ndarray, 
                    val_x: np.ndarray, 
                    val_batch: np.ndarray, 
                    val_bio: np.ndarray, 
                    num_epochs: int,
                    batch_size : int= 32):

        start = time.process_time()    

        self. hybrid_model.fit( train_x, 
                        {"de_output": train_x, "batch_output": train_batch,"bio_output":train_bio},
                        steps_per_epoch = len(train_x) // batch_size,
                        validation_data=(
                                        val_x,
                                        {"de_output": val_x, "batch_output": val_batch,"bio_output":val_bio}),
                        callbacks=[self.model_checkpoint_callback],
                        epochs=num_epochs,
                        verbose=0)

        print("Traing Time : ", time.process_time() - start)

        self.hybrid_model.load_weights('model.h5')

    def get_beene_embeddings(self, data: np.ndarray):
        return self.encoder_model.predict(data)

    def get_iLisi(self, data: np.ndarray, batch_var: np.ndarray ):
        """
            The method calulates the iLISI metric on the data using the beene embedding.

            Parametrs:

            data = 2D numpy array. The data matrix. The genes are along the columns and each row represents a sample
            batch_var = 1D numpy array.  The batch information for each of the samples.
        """

        if self.encoder_model == None:
            raise Exception("BEENE model is not created")

        batch_df = pd.DataFrame(batch_var,columns =['batch']) 

        beene_embeddings = self.encoder_model.predict(data)

        lisi_result = compute_lisi(beene_embeddings, batch_df, ['batch'])

        return lisi_result

    def evaluate_batch_iLisi(self, data: np.ndarray, batch_var: np.ndarray, bio_var: np.ndarray, seed: int ):

      """
      Compute the iLISI metric on the data. The data is split into testm training, and validation
      set. The model defined by the user is trained by the training set. The best performing model
      is selected by the validation set. And the iLISI index is calculated on the test set.

      return: iLISI values of each of the samples in the randomly choosen test set using BEENE embedding
      

      Parameters:

      data: 2D numpy array. Each row represents a sample and each column represents a Gene

      batch_var: numpy array. For more than two categories, it must be one hot representation of batch labels for each of the samples in the data and must be a dense matrix. For two categories, it must be a 1D array of zeros and ones denoting batch association for each samples in the data

      bio_var: numpy array. For more than two categories, it must be one hot representation of batch labels for each of the samples in the data and must be a dense matrix. For two categories,it must be a 1D array of zeros and ones denoting the biological class for each samples in the data

      seed: int. Random state for the split
      """

      # creating place holder for bio-var if null for traing the model

      if bio_var is None:
        bio_var = np.zeros_like(batch_var)
    
        data = data.astype('float64')
        batch_var = batch_var.astype('float64')
        bio_var = bio_var.astype('float64')
      

      X_train, X_test, Y_Platform_train, Y_Platform_test,Y_ER_train,Y_ER_test = train_test_split(
                                          data, batch_var, bio_var,test_size=0.20,random_state=seed)

      #Getting separate validation data

      X_train, X_val, Y_Platform_train, Y_Platform_val,Y_ER_train,Y_ER_val = train_test_split(
                                           X_train, Y_Platform_train, Y_ER_train ,test_size=0.25,random_state=seed)

      self.train_model(X_train,Y_Platform_train,Y_ER_train,X_val,Y_Platform_val,Y_ER_val,300)

      if  Y_Platform_test.ndim >1:
        Y_Platform_test = [np.argmax(i) for i in Y_Platform_test]
      ilisi_value = self.get_iLisi(X_test,Y_Platform_test)


      return ilisi_value






        

    
