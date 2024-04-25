import matplotlib.pyplot as plt


def freq_vs_cholestrol(df):

    # Extract the cholesterol levels of individuals with and without heart disease
    chol_hd = df[df['target'] == 1]['chol']
    chol_no_hd = df[df['target'] == 0]['chol']

    # Plot the distribution of cholesterol levels for each group
    plt.hist([chol_hd, chol_no_hd], bins=20, label=['Heart Disease', 'No Heart Disease'])
    plt.xlabel('Cholesterol Level')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Cholesterol Levels by Heart Disease Status')
    plt.show()