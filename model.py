import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
data=pd.read_csv('India_Flight.csv')
data.head()
x=data.drop(['Price','Unnamed: 0'],axis=1)
y=data['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=1)
params={'n_estimators':[i for i in np.arange(1,121)]}
from sklearn.model_selection import RandomizedSearchCV
model=RandomizedSearchCV(estimator=XGBRegressor(),param_distributions=params,cv=10)
model.fit(x_train,y_train)
model.score(x_test,y_test)

import pickle 
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))