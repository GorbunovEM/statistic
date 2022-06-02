import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer

#get info from data
data = pd.DataFrame(pd.read_csv('titanic.csv'))
data.info()
data.isna().any()

#fill NA
data['Cabin'] = data['Cabin'].fillna('01')

data['Age'] = data['Age'].fillna(data['Age'].median())
#or
data['Age'] = data['Age'].fillna(data['Age'].value_counts()[0])

data['Embarked'] = data.Embarked.fillna(data.Embarked.value_counts().index[0])

data['Fare'][data['Fare'].isna() == True] = np.where(data['Pclass'] == 1, data.groupby(['Pclass'], as_index=False).median().iloc[0]['Fare'], 
                                            np.where(data['Pclass'] == 2, data.groupby(['Pclass'], as_index=False).median().iloc[1]['Fare'],
                                                    data.groupby(['Pclass'], as_index=False).median().iloc[2]['Fare']))

#fill NA with KNN
imputer = KNNImputer()
imputer.fit()
new_data = imputer.transform()

#standardization
def z_score_standardization(series):
    return (series - series.mean()) / series.std())

data['Fare'] = z_score_standardization(data['Fare'])
data['Age'] = pd.DataFrame(np.array(normalize([data['Age']])[0]), columns=['Fare'])

#Label Encoder

le = LabelEncoder()
le.fit(data['Sex'].values)
le.classes_
data['Sex_transform'] = le.transform(data['Sex'].values)

#OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit_transform(data['Embarked'].values.reshape(-1,1))

#drop outlier
sns.boxplot(x = data['Age'])
z = np.abs(stats.zscore(data['Age']))
np.where(z > 3)


