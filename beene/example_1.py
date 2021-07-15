from beene import beene_model
from numpy import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np




# creating Random data with 3000 samples and 100 genes per sample.
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
# batch_weight: 2,
# bio_weight: 2,

my_model = beene_model()
my_model.get_hybrid_model_1(100,[50,20],5,3,2,1,2,1)



# Creating one hot vectors for batch variables

bt = np.reshape(bt,(-1,1))
enc_bi = OneHotEncoder(handle_unknown='ignore')
enc_bi.fit(bt)
bt = enc_bi.transform(bt)
bt = bt.todense()

# Number of classes in biological variables is 2. 
# So creating one-hot vector is not necessery


# calculating iLisi values for the data
lisi_values = my_model.evaluate_batch_iLisi(Xt,bt,yt,20)

print(np.median(lisi_values))


