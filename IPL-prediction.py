
# coding: utf-8

# In[255]:


# Import numpy and pandas library
import pandas as pd
import numpy as np

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

# Input training and test data
deliveries = pd.read_csv("Data/deliveries.csv")
match = pd.read_csv("Data/matches.csv")
IPL_2018 = pd.read_csv("Data/test_data.csv")
# Choice of fields for current model from available data
matches = match[["city","venue","season","toss_winner","toss_decision", "team1", "team2", "winner"]]


# In[256]:


# Addition of customized field
match_by_ball = deliveries[["match_id", "inning", "batsman","batsman_runs", "extra_runs", "total_runs"]]

match_by_ball = match_by_ball[match_by_ball["inning"] == 1]
match_by_ball.drop(["inning"], axis=1, inplace=True)

extra_stats = match_by_ball.groupby(["match_id", "batsman"])["batsman_runs", "extra_runs", "total_runs"].sum()

extra_stats["half_century"] = extra_stats["batsman_runs"] >= 50

newdf = extra_stats.groupby("match_id").sum().drop("batsman_runs",1).reset_index(drop = True)
matches = pd.concat([matches, newdf], axis=1)


# In[257]:


# Choice of fields for test data
IPL_2018 = IPL_2018[["city","venue","season","toss_winner","toss_decision", "team1", "team2", "winner", "total_runs", "half_century", "extra_runs"]]


# In[258]:


# Merging training and test data for preprocessing
matches = pd.concat([IPL_2018, matches]).reset_index(drop=True)


# In[259]:


# Converting data into factors
[factors, matches] = factorize_fields(matches)

# Removing season: Include if time model made
matches.drop(["season"],1,inplace=True)


# In[260]:


# Deriving insight on data
matches.describe()


# In[261]:


# Train - test split after pre processing
ind = len(IPL_2018)
test_data = matches[:ind]
train_data = matches[ind:]


# In[262]:


def remap(item):
    '''Function to remap the factors to original field names'''
    return factors[item]


# In[263]:


def model_building(model, predictors, outcome, data, test_data):
    '''Function to build model, cross-validate and predict results'''
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


# In[264]:


# Models tested
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


# In[265]:


# Final result
model_comparison
