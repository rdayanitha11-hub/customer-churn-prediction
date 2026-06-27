import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
df = pd.read_csv("data/archive/Telco-Customer-Churn.csv")