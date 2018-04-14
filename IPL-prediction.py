
# coding: utf-8

# In[133]:


import pandas as pd
import numpy as np

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

deliveries = pd.read_csv("Data/deliveries.csv")
match = pd.read_csv("Data/matches.csv")
matches = match[["city","venue","season","toss_winner","toss_decision","team1","team2","winner"]]
#matches.head()


# In[134]:


def factorize_fields(matches):
    df = matches[["team1","team2","toss_winner","winner"]]
    _, b = pd.factorize(df.values.T.reshape(-1, ))  
    facorized_fields = df.apply(lambda x: pd.Categorical(x, b).codes)
    matches["venue_"] = matches.venue.factorize()[0]
    matches["city_"] = matches.city.factorize()[0]
    matches["toss_winner_"] = matches.toss_winner.factorize()[0]
    matches["toss_decision_"] = matches.toss_decision.factorize()[0]
    matches = pd.concat([matches[["city_","venue_","season","toss_winner_","toss_decision_"]], facorized_fields], 1)
    factors = pd.factorize(df.values.T.reshape(-1, ))[1]
    return [factors, matches]


# In[135]:


[factors, matches] = factorize_fields(matches)


# In[136]:


# Removing season: Include if time model made
matches.drop(["season"],1,inplace=True)


# In[137]:


def remap(item):
    return factors[item]


# In[138]:


def model_building(model, predictors, outcome, data, test_data):
    model.fit(data[predictors], data[outcome])  
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    cv_error = np.mean(error)
    #print('Cross validation Score : %s' % '{0:.3%}'.format(cv_error))
    model.fit(data[predictors],data[outcome])
    predictions = model.predict(test_data[predictors])
    test_data["predicted_winner"] = predictions
    accuracy = metrics.accuracy_score(predictions,test_data[outcome])
    #print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
    test_data["Team1"] = test_data["team1"].apply(remap)
    test_data["Team2"] = test_data["team2"].apply(remap)
    test_data["Actual Winner"] = test_data["winner"].apply(remap)
    test_data["Predicted Winner"] = test_data["predicted_winner"].apply(remap)
    df = test_data[["Team1","Team2","Actual Winner", "Predicted Winner"]]
    return [df, accuracy, cv_error]


# In[139]:


# Temporary train-test split: to be removed after obtaining the actual test data
ind = int(0.2*len(matches))
test_data = matches[:ind]
train_data = matches[ind:]


# In[144]:


model1 = RandomForestClassifier(n_estimators=100)
model2 = LogisticRegression()
model3 = SVC()
models = [model1, model2, model3]
results = []
accuracies = []
cv_errors = []
for model in models:
    output = ['winner']
    predictors = matches.drop(["winner"],1).columns
    [result, accuracy, cv_error] = model_building(model, predictors, output, train_data, test_data)
    results.append(result)
    accuracies.append(accuracy)
    cv_errors.append(cv_error)
model_names = ["RandomForest", "LogisticRegression", "SVM"]
model_comparison = pd.DataFrame(columns=["Model Names", "Accuracy", "Cross Validation Errors"])
model_comparison["Model Names"] = model_names
model_comparison["Accuracy"] = accuracies
model_comparison["Cross Validation Errors"] = cv_errors


# In[145]:


model_comparison

