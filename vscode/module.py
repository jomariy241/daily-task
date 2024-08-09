from sklearn.datasets import load_iris
from sklearn .model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=RandomForestClassifier(random_state=42)
clf.fit(x_train,y_train)

import pickle
with open('model.pkl','wb')as model_file:
  pickle.dump(clf,model_file)