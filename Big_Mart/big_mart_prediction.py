import pandas as pd
import numpy as np

test = pd.read_csv('test_big_mart.csv')
train = pd.read_csv('train_big_mart.csv')

train['source'] = 'train'
test['source'] = 'test'
#Combining dataframes to perform feature engineering
data = pd.concat([train, test], ignore_index=True)

#check missing values
is_null = data.apply(lambda x: sum(x.isnull()))
data_des = data.describe()

#looking at the number of unique values 
u_values = data.apply(lambda x: len(x.unique()))

#Exploring the frequencie of different categories
cat_column = [x for x in data.dtypes.index if data.dtypes[x] == 'object']
cat_column = [x for x in cat_column if x not in ['Item_Identifier','Outlet_Identifier','source']]

for col in cat_column:
    print('\nFrequency of Categories for variable %s'%col)
    print(data[col].value_counts())
    
#Inputting missing values into Item_Weight using avergae weight
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
miss_bool = data['Item_Weight'].isnull()

#Confirming mising data was inputted
print('Original missing data in item weight: %d' %sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x, 'Item_Weight'])
print('Final missing data: %d' %sum(data['Item_Weight'].isnull()))
    
#Replace missing values of Outlet_Size with the Mode
data['Outlet_Size'].fillna('Medium', inplace=True)

#Looking at correlation between supermarket type 2 and type 3
data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')

#Determine average visibility of a product and input it for the 0 values
vis_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
miss_bool = (data['Item_Visibility'] == 0)
data.loc[miss_bool, 'Item_Visibility'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: vis_avg.at[x, 'Item_Visibility'])
print(sum(data['Item_Visibility'] == 0))

#Combining ItemType
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food',
    'NC' : 'Non-Consumable',
    'DR': 'Drinks'})

#Making column based on years in operation of the store
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

#Fixing inconsistencies with Item_Fat_Content
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
    'reg' : 'Regular',
    'low fat' : 'Low Fat'})

data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

#Encoding categorial variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type'])

#Exporting dataframes back to their sets

data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)

#Model Building
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']

from sklearn import metric
from sklearn.model_selection import train_test_split

def model(alg, dtrain, dtest, predictors, target, IDcol, filename):
    
    #Fitting model on the data
    alg.fit(dtrain[predictors], dtrain[target])
    
    #Predict Training Set
    dtrain_pred = alg.predict(dtrain[predictors])
    
    #Perform Cross Validation
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    
    cv_score = np.sqrt(np.abs(cv_score))
    
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
      
    #Predict on testing data
    dtest[target] = alg.predict(dtest[predictors])
    
    
from sklearn.linear_model import LinearRegression, Ridge, Lasso

predictors = [x for x in train.columns if x not in [target] + IDcol]  
alg1 = LinearRegression(normalize=True)
model(alg1, train, test, predictors, target, IDcol, 'alg0.csv')
    
    
    
    
    
    
    
    
    
    
    




























