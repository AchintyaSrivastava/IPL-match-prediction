
# coding: utf-8

# In[22]:


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
matches = match[["city","venue","season","toss_winner","toss_decision", "result", "dl_applied", "team1", "team2", "winner"]]


# In[23]:


# Addition of customized field
match_by_ball = deliveries[["match_id", "inning", "batsman","batsman_runs", "extra_runs", "total_runs"]]

match_by_ball = match_by_ball[match_by_ball["inning"] == 1]
match_by_ball.drop(["inning"], axis=1, inplace=True)

extra_stats = match_by_ball.groupby(["match_id", "batsman"])["batsman_runs", "extra_runs", "total_runs"].sum()

extra_stats["half_century"] = extra_stats["batsman_runs"] >= 50

newdf = extra_stats.groupby("match_id").sum().drop("batsman_runs",1).reset_index(drop = True)
matches = pd.concat([matches, newdf], axis=1)


# In[24]:


# Choice of fields for test data
IPL_2018 = IPL_2018[["city","venue","season","toss_winner","toss_decision", "result", "dl_applied", "team1", "team2", "winner", "total_runs", "half_century", "extra_runs"]]


# In[25]:


# Removing matches resulting in tie, having no results or being finalized via D/L method
matches = matches[matches["result"] != "tie"]
matches = matches[matches["result"] != "no result"]
matches = matches[matches["dl_applied"] == 0]
IPL_2018 = IPL_2018[matches["result"] != "tie"]
IPL_2018 = IPL_2018[matches["result"] != "no result"]
IPL_2018 = IPL_2018[matches["dl_applied"] == 0]


# In[26]:


# Merging training and test data for preprocessing
matches = pd.concat([IPL_2018, matches]).reset_index(drop=True)

matches.drop(["dl_applied", "result"],1,inplace=True)


# In[27]:


def factorize_fields(matches):
    df = matches[["team1","team2","toss_winner","winner"]]
    _, b = pd.factorize(df.values.T.reshape(-1, ))  
    facorized_fields = df.apply(lambda x: pd.Categorical(x, b).codes)
    matches["venue"] = matches.venue.factorize()[0]
    matches["city"] = matches.city.factorize()[0]
    matches["toss_winner"] = matches.toss_winner.factorize()[0]
    matches["toss_decision"] = matches.toss_decision.factorize()[0]
    matches = pd.concat([matches.drop(["team1","team2","toss_winner","winner"],1), facorized_fields], 1)
    factors = pd.factorize(df.values.T.reshape(-1, ))[1]
    return [factors, matches]
#[["city","venue","season","toss_winner","toss_decision"]]


# In[28]:


# Converting data into factors
[factors, matches] = factorize_fields(matches)

# Removing season: Include if time model made
matches.drop(["season"],1,inplace=True)


# In[29]:


# Deriving insight on data
matches.describe()


# In[30]:


# Train - test split after pre processing
ind = len(IPL_2018)
test_data = matches[:ind]
train_data = matches[ind:]


# In[31]:


def remap(item):
    '''Function to remap the factors to original field names'''
    return factors[item]


# In[32]:


def model_building(model, predictors, outcome, data, test_data):
    '''Function to build model, cross-validate and predict results'''
    #model.fit(data[predictors], data[outcome])  
    kf = KFold(data.shape[0], n_folds = 3)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    cv_error = np.mean(error)
    #print('Cross validation Score : %s' % '{0:.3%}'.format(cv_error))
    model.fit(data[predictors],data[outcome])
    #coefficients = [model.intercept_, model.coef_]
    #print coefficients
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


# In[33]:


# Models tested
model1 = RandomForestClassifier(n_estimators=100)
model2 = LogisticRegression()         #L2 regularization, one vs all
model3 = LogisticRegression(penalty='l1')         #L1 regularization, one vs all
model4 = LogisticRegression(solver='newton-cg', multi_class='multinomial')  #Multinomial
model5 = SVC(kernel = "linear")
models = [model1, model2, model3, model4, model5]
results = []
accuracies = []
cv_errors = []
#col = ["Intercept"] + list(train_data.columns)
#coefficients_summary = pd.DataFrame(columns= col)
output = ['winner']
predictors = matches.drop(["winner"],1).columns
for model in models:
    [result, accuracy, cv_error] = model_building(model, predictors, output, train_data, test_data)
    results.append(result)
    accuracies.append(accuracy)
    cv_errors.append(cv_error)
    #coefficients_summary = coefficients_summary.append(coefficients, ignore_index=True)
model_names = ["Random Forest", "LogisticRegression(One vs All) L2 reg", "LogisticRegression(One vs All) L1 reg", "MultinomialRegression", "SVM"]
model_comparison = pd.DataFrame(columns=["Model Names", "Accuracy", "Cross Validation Errors"])
model_comparison["Model Names"] = model_names
model_comparison["Accuracy"] = accuracies
model_comparison["Cross Validation Errors"] = cv_errors


# In[34]:


# Final result
model_comparison


# In[35]:


# Post model fit analysis
import statsmodels.formula.api as sm
model = sm.MNLogit(train_data[output], train_data[predictors]) 
mod = model.fit()


# In[36]:


print mod.summary()

