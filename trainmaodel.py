import pandas as pd
dataset=pd.read_csv('Crop_recommendation.csv')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle

x=dataset.iloc[:, [0,1,2,3,4,5,6]].values
y=dataset.iloc[:, [7]].values

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,)

classifier=KNeighborsClassifier(n_neighbors=3)

ytrain=np.ravel(ytrain)

classifier.fit(xtrain,ytrain)

file=open('cropmodel.pkl','wb')
pickle.dump(classifier,file)
file.close()


