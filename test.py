import numpy as np
import pandas as pd
import datetime

df = pd.read_csv('train.csv')

print(df.describe())

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

#Check to see which values have missing data
print(df.apply(lambda x: sum(x.isnull()), axis=0))


#Filling in missing values in Loan Amount
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

#Looking at distribution of loan amounts between self-employed/employed and education
print(df.boxplot(['LoanAmount'], by=['Education', 'Self_Employed']))

#Filling in missing values for Self Employed, Gender,Dependents, Married, Credit History and Loan Amount Term
df['Self_Employed'].fillna('No', inplace=True)
df['Gender'].fillna('Male', inplace=True)
df['Dependents'].fillna('0', inplace=True)
df['Married'].fillna('Yes', inplace=True)
df['Credit_History'].fillna('1.0', inplace=True)
df['Loan_Amount_Term'].fillna('360.0', inplace=True)

#Making Pivot table to examine values of the loan amount 
table = df.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)

def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]

df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

#Normalizing Loan Amounts
df['LoanAmount_log'] = np.log(df['LoanAmount'])

#Combining applicant and co-applicant income 
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'] =np.log(df['TotalIncome'])

from sklearn.preprocessing import LabelEncoder
#Encoding all string variables into numerical values for sklearn
var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
    
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

def classification_model(model, data, predictors, outcome):
    
    #Fitting the model
    model.fit(data[predictors], data[outcome])
    
    #Make predictions on training set
    predictions = model.predict(data[predictors])
    
    #Printing accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accurary : %s" % "{0:3%}".format(accuracy))
    
    #Perform K-Fold cross_validation using 5 folds
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(data.shape[0]):
        train_predictors = (data[predictors].iloc[train, :])
        
        train_target = data[outcome].iloc[train]
        
        model.fit(train_predictors, train_target)
    
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
        
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
         
        model.fit(data[predictors], data[outcome])
        
        
#Making first model for Credit_History based prediction
outcome_var = 'Loan_Status'
model = LogisticRegression()
prediction_var = ['Credit_History']
classification_model(model, df, prediction_var, outcome_var)












