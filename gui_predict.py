import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk

from sklearn.svm import SVC



url = 'datasets/processed.cleveland.data.csv'
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


root = tk.Tk()
root.title('Heart Disease Prediction')

def focus_next_widget(event):
    event.widget.tk_focusNext().focus()
    return "break"

def focus_previous_widget(event):
    event.widget.tk_focusPrev().focus()


# Age
tk.Label(root, text='Age:').grid(row=0, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1)
age_entry.bind("<Return>", focus_next_widget)
age_entry.bind("<Up>", focus_previous_widget)
age_entry.bind("<Down>", focus_next_widget)


# Sex
tk.Label(root, text='Sex (0 for female, 1 for male):').grid(row=1, column=0)
sex_entry = tk.Entry(root)
sex_entry.grid(row=1, column=1)
sex_entry.bind("<Return>", focus_next_widget)
sex_entry.bind("<Up>", focus_previous_widget)
sex_entry.bind("<Down>", focus_next_widget)

# Chest pain type
tk.Label(root, text='Chest pain type (1-4):').grid(row=2, column=0)
cp_entry = tk.Entry(root)
cp_entry.grid(row=2, column=1)
cp_entry.bind("<Return>", focus_next_widget)
cp_entry.bind("<Up>", focus_previous_widget)
cp_entry.bind("<Down>", focus_next_widget)

# Resting blood pressure
tk.Label(root, text='Resting blood pressure:').grid(row=3, column=0)
trestbps_entry = tk.Entry(root)
trestbps_entry.grid(row=3, column=1)
trestbps_entry.bind("<Return>", focus_next_widget)
trestbps_entry.bind("<Up>", focus_previous_widget)
trestbps_entry.bind("<Down>", focus_next_widget)

# Serum cholesterol
tk.Label(root, text='Serum cholesterol in mg/dl:').grid(row=4, column=0)
chol_entry = tk.Entry(root)
chol_entry.grid(row=4, column=1)
chol_entry.bind("<Return>", focus_next_widget)
chol_entry.bind("<Up>", focus_previous_widget)
chol_entry.bind("<Down>", focus_next_widget)

# Fasting blood sugar
tk.Label(root, text='Fasting blood sugar > 120 mg/dl (0 for no, 1 for yes):').grid(row=5, column=0)
fbs_entry = tk.Entry(root)
fbs_entry.grid(row=5, column=1)
fbs_entry.bind("<Return>", focus_next_widget)
fbs_entry.bind("<Up>", focus_previous_widget)
fbs_entry.bind("<Down>", focus_next_widget)

# Resting electrocardiographic results
tk.Label(root, text='Resting electrocardiographic results (0-2):').grid(row=6, column=0)
restecg_entry = tk.Entry(root)
restecg_entry.grid(row=6, column=1)
restecg_entry.bind("<Return>", focus_next_widget)
restecg_entry.bind("<Up>", focus_previous_widget)
restecg_entry.bind("<Down>", focus_next_widget)

# Maximum heart rate achieved
tk.Label(root, text='Maximum heart rate achieved:').grid(row=7, column=0)
thalach_entry = tk.Entry(root)
thalach_entry.grid(row=7, column=1)
thalach_entry.bind("<Return>", focus_next_widget)
thalach_entry.bind("<Up>", focus_previous_widget)
thalach_entry.bind("<Down>", focus_next_widget)

# Exercise-induced angina
tk.Label(root, text='Exercise-induced angina (0 for no, 1 for yes):').grid(row=8, column=0)
exang_entry = tk.Entry(root)
exang_entry.grid(row=8, column=1)
exang_entry.bind("<Return>", focus_next_widget)
exang_entry.bind("<Up>", focus_previous_widget)
exang_entry.bind("<Down>", focus_next_widget)

# ST depression induced by exercise
tk.Label(root, text='ST depression induced by exercise relative to rest (0-6):').grid(row=9, column=0)
oldpeak_entry = tk.Entry(root)
oldpeak_entry.grid(row=9, column=1)
oldpeak_entry.bind("<Return>", focus_next_widget)
oldpeak_entry.bind("<Up>", focus_previous_widget)
oldpeak_entry.bind("<Down>", focus_next_widget)

tk.Label(root, text='Slope of the peak exercise ST segment (0-2):').grid(row=10, column=0)
slope_entry = tk.Entry(root)
slope_entry.grid(row=10, column=1)
slope_entry.bind("<Return>", focus_next_widget)
slope_entry.bind("<Up>", focus_previous_widget)
slope_entry.bind("<Down>", focus_next_widget)

tk.Label(root, text='Number of major vessels colored by flourosopy (0-3):').grid(row=11, column=0)
ca_entry = tk.Entry(root)
ca_entry.grid(row=11, column=1)
ca_entry.bind("<Return>", focus_next_widget)
ca_entry.bind("<Up>", focus_previous_widget)
ca_entry.bind("<Down>", focus_next_widget)

tk.Label(root, text='Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect):').grid(row=12, column=0)
thal_entry = tk.Entry(root)
thal_entry.grid(row=12, column=1)
thal_entry.bind("<Return>", focus_next_widget)
thal_entry.bind("<Up>", focus_previous_widget)
thal_entry.bind("<Down>", focus_next_widget)

def predict(event=None):
    # Get input values
    age = float(age_entry.get())
    sex = float(sex_entry.get())
    cp = float(cp_entry.get())
    trestbps = float(trestbps_entry.get())
    chol = float(chol_entry.get())
    fbs = float(fbs_entry.get())
    restecg = float(restecg_entry.get())
    thalach = float(thalach_entry.get())
    exang = float(exang_entry.get())
    oldpeak = float(oldpeak_entry.get())
    slope = float(slope_entry.get())
    ca = float(ca_entry.get())
    thal = float(thal_entry.get())

    # Create input array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])

    input_data = scaler.transform(input_data)

    target = clf.predict(input_data)[0]

    if target == 0:
        result_label.config(text='No Heart Disease')
    else:
        result_label.config(text='Heart Disease Present')

predict_button = tk.Button(root, text='Predict', command=predict)
predict_button.grid(row=13, column=0, columnspan=2)
predict_button.bind("<Return>", predict)

result_label = tk.Label(root, text='')
result_label.grid(row=14, column=0, columnspan=2)

root.mainloop()






