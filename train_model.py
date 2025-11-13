import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("house_price.csv")

X = df.iloc[:,0:2]
y = df["price"]


X_train, X_test, y_trian, y_test = train_test_split(X,y,train_size=0.25,random_state=0)

lrmodel = LinearRegression()

lrmodel.fit(X_train,y_trian)

with open("lrmodel.pkl","wb") as f:
    pickle.dump(lrmodel,f)

print("Model trained and saved as lrmodel.pkl")