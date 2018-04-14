import pandas as pd
import numpy as np

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

deliveries = pd.read_csv("Data/deliveries.csv")
matches = pd.read_csv("Data/matches.csv")

df = matches[["team1","team2","toss_winner","winner"]]
#print df.head()
_, b = pd.factorize(df.values.T.reshape(-1, ))  
facorized_fields = df.apply(lambda x: pd.Categorical(x, b).codes)
#df1 = pd.concat([df, r], 1)
#print r.head()
#print matches.head()
matches["venue_"] = matches.venue.factorize()[0]
#matches.drop(["venue"], axis=1,inplace=True)
matches["city_"] = matches.city.factorize()[0]
matches["toss_winner_"] = matches.toss_winner.factorize()[0]
matches["toss_decision_"] = matches.toss_decision.factorize()[0]
matches = pd.concat([matches[["city_","venue_","season","toss_winner_","toss_decision_"]], facorized_fields], 1)
print matches.head()


def model_building(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
  print('Cross validation Score : %s' % '{0:.3%}'.format(np.mean(error)))
  model.fit(data[predictors],data[outcome])
  return model

model = RandomForestClassifier(n_estimators=100)
output = ['winner']
predictors = matches.drop(["winner"]).columns#['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
model = model_building(model, matches, predictors, output)
