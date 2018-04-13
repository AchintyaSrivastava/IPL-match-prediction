import pandas as pd
import numpy as np

deliveries = pd.read_csv("Data/deliveries.csv")
matches = pd.read_csv("Data/matches.csv")

team1 = "Mumbai Indians"
team2 = "Royal Challengers Bangalore"

MI = matches[matches["team1"] == team1].append(matches[matches["team2"] == team1])
CSK = matches[matches["team1"] == team2].append(matches[matches["team2"] == team2])
print MI.describe()
print CSK.describe()