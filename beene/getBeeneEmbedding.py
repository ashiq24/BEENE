# Importing necessery packages

from beene import beene_model
from numpy import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split



# creating Random data with 3000 sample and 100 genes per sample.
Xt = random.uniform(-1,1,(3000,100))
# With 2 biological categories 
yt = random.randint(0,2,3000)
# With 3 batches
bt = random.randint(0,3,3000)

# Creating the BEENE model
# with embedding dimension of 5
# Size of first hiddent layer is 50, and second hidden
# layer is 20
# reconstruction_weight: 1,
# batch_prediction_loss_weight: 2,
# biovar_prediction_loss_weight: 2,

my_model = beene_model()
my_model.get_hybrid_model_1(100,[50,20],5,3,2,1,2,1)

# Creating one hot vectors for batch variables
# Number of classes in biological variables is 2. 
# So creating one-hot vector is not necessery

bt = np.reshape(bt,(-1,1))
enc_bi = OneHotEncoder(handle_unknown='ignore')
enc_bi.fit(bt)
bt = enc_bi.transform(bt)
bt = bt.todense()


###
# Creating training-validation-test split
###
X_train, X_test, Y_Platform_train, Y_Platform_test,Y_ER_train,Y_ER_test = train_test_split(
                                          Xt, bt, yt,test_size=0.20,random_state=4)

#Getting separate validation data
X_train, X_val, Y_Platform_train, Y_Platform_val,Y_ER_train,Y_ER_val = train_test_split(
                                           X_train, Y_Platform_train, Y_ER_train ,test_size=0.25,random_state=4)

# Training the model for 300 epochs

my_model.train_model(X_train,Y_Platform_train,Y_ER_train,X_val,Y_Platform_val,Y_ER_val,100)

##
# Saving embedding for test set
# Embeddings will be saved in txt format. Emebedings for each of the cells are stored along
# the rows. Values are space separated. 
##

test_embedding = my_model.get_beene_embeddings(X_test)

np.savetxt('embedding.txt', test_embedding, fmt='%f')

## for loading 

loaded_embedding = np.loadtxt('embedding.txt', dtype=float)