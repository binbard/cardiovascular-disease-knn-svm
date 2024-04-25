import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC


# Load the dataset
url = 'datasets/processed.cleveland.data.csv'
df = pd.read_csv(url, header=None)

# Set column names
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
opt = "svm"
# opt = input("Enter the classifier to use [knn, svm, rf, dt]: ")

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


# Take input from the user
age = float(input('Enter age: '))
sex = float(input('Enter sex (0 for female, 1 for male): '))
cp = float(input('Enter chest pain type (1-4): '))
trestbps = float(input('Enter resting blood pressure: '))
chol = float(input('Enter serum cholesterol in mg/dl: '))
fbs = float(input('Enter fasting blood sugar > 120 mg/dl (0 for no, 1 for yes): '))
restecg = float(input('Enter resting electrocardiographic results (0-2): '))
thalach = float(input('Enter maximum heart rate achieved: '))
exang = float(input('Enter exercise-induced angina (0 for no, 1 for yes): '))
oldpeak = float(input('Enter ST depression induced by exercise relative to rest: '))
slope = float(input('Enter the slope of the peak exercise ST segment (1-3): '))
ca = float(input('Enter number of major vessels (0-3) colored by fluoroscopy: '))
thal = float(input('Enter thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect): '))

# Make a prediction
x_new = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
x_new = scaler.transform(x_new)
y_new = clf.predict(x_new)

# Display the result
if y_new == 0:
    print('The model predicts that the patient does not have heart disease.')
else:
    print('The model predicts that the patient has heart disease.')


'''
Sample data:
Have heart disease: [56,1,3,130,256,0,2,142,1,0.6,2,1,3]
Does not have: [40,1,3,120,200,0,0,160,0,0.5,2,0,3]
'''