import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

data = pd.DataFrame(pd.read_csv('Crop_recommendation.csv'))
data.label = pd.Categorical(data.label)
X = data.drop('label',axis=1)
Y = pd.Categorical(data.label)
Y_c = Y.codes

x_train, x_test, y_train, y_test = train_test_split(X,Y_c,test_size=0.2)
sns.scatterplot(x=data.temperature, y=data.humidity, hue=data.label, data=data)
plt.show()

(data == 0).sum(axis=1)
data.isna().sum(axis=1)

clf = DecisionTreeClassifier(random_state=0)
cross_val_score(clf, x_train, y_train)
model_1 = clf.fit(x_train, y_train)

max_depth_range = list(range(1, 20))
accuracy = []
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
    clf.fit(x_train, y_train)    
    score = clf.score(x_test, y_test)
    accuracy.append(score)
    
plt.plot(range(1, 20),accuracy)
plt.show
