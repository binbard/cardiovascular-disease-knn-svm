import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def k_vs_score(X_train,y_train,X_test,y_test):
    # Vary the value of k from 1 to 30
    k_range = range(1, 31)

    # Calculate the accuracy score for each value of k
    scores = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)

    # Plot the results in a line chart
    plt.plot(k_range, scores)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy Score')
    plt.title('KNN Classifier Accuracy')
    plt.show()