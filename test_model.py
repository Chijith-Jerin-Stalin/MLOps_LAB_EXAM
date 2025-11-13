import pandas as pd

df = pd.read_csv("Heart.csv")


X = df.iloc[:,:-1]


y = df["Thal"]
