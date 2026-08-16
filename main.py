import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle as pkl


# Load and prepare the data 
read_csv = pd.read_csv(r"data_set\UCI_Credit_Card.csv")
read_csv = read_csv.sample(3000,random_state=42)
print(read_csv.head())
print(read_csv.describe())
print(read_csv.info())


# Filtering data for prediction model
find_missing_value = read_csv.isnull().sum()
find_duplicate_value = read_csv.duplicated().sum()
print(find_missing_value)
print(find_duplicate_value)

# Drop columns
read_csv.drop(["ID","MARRIAGE","EDUCATION"],axis=1,inplace=True)


# Split data for traing and testing
X = read_csv.drop("default.payment.next.month",axis=1)
y = read_csv["default.payment.next.month"]
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
print(x_train)


# # Now check is the data is balanced or unbalanced
print(y_train.value_counts())

# Data is unbalance so we use sampling techniques
sampler = SMOTE(k_neighbors = 10)
x_train_sampler,y_train_sampler = sampler.fit_resample(x_train,y_train)
print(x_train_sampler)

# Now again chk data Balance
print(y_train_sampler.value_counts())

# Scaling the data 
scaler = MinMaxScaler()
x_train_scaler = scaler.fit_transform(x_train_sampler) 
x_test_scaler = scaler.transform(x_test) 
print(x_test_scaler)

# Using LogisticRegression Model 
lr_model = LogisticRegression()
lr_model.fit(x_train_scaler,y_train_sampler)
lr_pred = lr_model.predict(x_test_scaler)

# Chk model accuray 
print(f"Model Accuracy : {accuracy_score(y_test,lr_pred)}")
print(f"Classification Report : {classification_report(y_test,lr_pred)}")
print(f"Confusion Matrix : {confusion_matrix(y_test,lr_pred)}")


# Using RandomForestClassifier Model 
rfc_model = RandomForestClassifier()
rfc_model.fit(x_train_scaler,y_train_sampler)
rfc_pred = rfc_model.predict(x_test_scaler)

# Chk model accuray 
print(f"Model Accuracy : {accuracy_score(y_test,rfc_pred)}")
print(f"Classification Report : {classification_report(y_test,rfc_pred)}")
print(f"Confusion Matrix : {confusion_matrix(y_test,rfc_pred)}")


# Using DecisionTreeClassifier Model 
dtc_model = DecisionTreeClassifier()
dtc_model.fit(x_train_scaler,y_train_sampler)
dtc_pred = dtc_model.predict(x_test_scaler)

# Chk model accuray 
print(f"Model Accuracy : {accuracy_score(y_test,dtc_pred)}")
print(f"Classification Report : {classification_report(y_test,dtc_pred)}")
print(f"Confusion Matrix : {confusion_matrix(y_test,dtc_pred)}")

# Using VotingClassifier Model 
pred_model = VotingClassifier(estimators = [("LogisticRegression",lr_model),("RandomForestClassifier",rfc_model)] , voting="hard")
pred_model.fit(x_train_scaler,y_train_sampler)
pred_model_res = pred_model.predict(x_test_scaler)

# Chk model accuray
print(f"Model Accuracy : {accuracy_score(y_test,pred_model_res)}")
print(f"Classification Report : {classification_report(y_test,pred_model_res)}")
print(f"Confusion Matrix : {confusion_matrix(y_test,pred_model_res)}")

pkl.dump(rfc_model,open(r"Models\Model.pkl","wb"))
pkl.dump(scaler,open(r"Models\Scaler.pkl","wb"))