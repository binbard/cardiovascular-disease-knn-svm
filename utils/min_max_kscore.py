from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def min_max_score(X_train,y_train,X_test,y_test,y_pred):
    k_range = range(1, 31)

    # Calculate the accuracy score for each value of k
    scores = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)

    # Find the value of k at which the accuracy score is maximum and minimum
    max_k = k_range[scores.index(max(scores))]
    min_k = k_range[scores.index(min(scores))]

    print('Maximum accuracy score of', max(scores), 'achieved at k =', max_k)
    print('Minimum accuracy score of', min(scores), 'achieved at k =', min_k)