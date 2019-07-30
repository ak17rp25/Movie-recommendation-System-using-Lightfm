#MOVIE RECCOMMENDATION SYSTEM

#two types of recomandation system
#1.collabarative: what you like based on what other similar user liked
#2.content based: What you like based on what u liked on past

#numpy and scipy will allow us to some arithmetic work or math

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data=fetch_movielens(min_rating=3.0)
#min rating is the minimum rating we want to have in our model

#create model
model=LightFM(loss='warp')
model1=LightFM(loss='logistic')

#warp is weighted approprixamate rank pairwise

#it is hybrid system collabarive+content based

model.fit(data['train'],epochs=20,num_threads=2)
model1.fit(data['train'],epochs=20,num_threads=2)
def sample_recommendation(model,data,user_ids):
    #number of users and movies in our training set
    
    n_users,n_items=data['train'].shape
    #generate recommendation for each user we input
    
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        print("User %s" % user_id)
        print("     Known positives:")
        for x in known_positives[:3]:
            print("        %s" % x)
        print("     Recommended:")
        for x in top_items[:3]:
            print("        %s" % x)
print("USING LOSS WARP METHOD")
sample_recommendation(model, data, [3, 25, 450])
print("USING LOSS Logistic METHOD")
sample_recommendation(model1, data, [3, 25, 450])

