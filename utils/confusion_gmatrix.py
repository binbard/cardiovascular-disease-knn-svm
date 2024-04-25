from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def confusion_gmatrix(y_test, y_pred):
    
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a heatmap to visualize the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()