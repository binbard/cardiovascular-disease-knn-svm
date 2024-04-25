from sklearn.svm import SVC

from utils import score_vs_k
from utils import freq_cholestrol
from utils import confusion_gmatrix

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

url = 'datasets/heart_disease_dataset.csv'

df = pd.read_csv(url, header=None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
              'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Preprocess the data
df = df.replace('?', np.nan)
df = df.dropna()

# Split the data into training and testing sets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# opt = ["knn","svm","rf","dt"]
# opt = "knn"
opt = input("Enter the classifier to use [knn, svm, rf, dt]: ")

clf = None

if(opt == "knn"):
    # Train the SVM classifier
    k = 6
    clf = KNeighborsClassifier(n_neighbors=k)

elif(opt == "svm"):
    # Train the SVM classifier
    C = 1.0
    clf = SVC(kernel='linear', C=C)

elif(opt == "rf"):
    # Create the random forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

elif(opt == "dt"):
    # Create the decison tree classifier
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)

if clf is None:
    print("Invalid classifier")
    exit(1)
    
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# Show test data to the user
test_data = pd.DataFrame(X_test, columns=df.columns[:-1])
test_data['target'] = y_test
print("Test data:\n", test_data)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# Value of k at which accuracy is max and min ---
score_vs_k.k_vs_score(X_train,y_train,X_test,y_test)

# Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

# Freqency of cholestrol levels vs healty and unhealthy data ---
freq_cholestrol.freq_vs_cholestrol(df)

# Confusion Matrix +++
confusion_gmatrix.confusion_gmatrix(y_test, y_pred)


# Scatter matrix +++
pd.plotting.scatter_matrix(df, figsize=(15,15), diagonal='hist')
plt.show()


# Classification Report +++
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)
print('Accuracy:', accuracy)

