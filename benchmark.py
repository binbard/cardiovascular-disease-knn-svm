import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from utils import min_max_kscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


url = 'datasets/processed.cleveland.data.csv'
df = pd.read_csv(url, header=None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
              'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = df.replace('?', np.nan)
df = df.dropna()

# Split the data into training and testing sets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# opt = ["knn","svm","rf","dt"]
# opt = "knn"
opt = input("Enter the classifier to use [knn, svm, rf, dt]: ")

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


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)


if(opt == "knn"):
    print('Accuracy at k',str(k),'is', accuracy*100,'%')
    min_max_kscore.min_max_score(X_train, y_train, X_test, y_test, y_pred)

else:
    print('Accuracy using ' + opt + ' is ' + str(accuracy*100),'%')


