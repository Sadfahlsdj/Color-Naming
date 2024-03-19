"""Comments on csv:
This colour naming dataset was collected @colornaming.net and published in Mylonas, D., &
MacDonald, L. (2016). Augmenting basic colour terms in English. Color Research & Application,
41(1), 32�42. https://doi.org/10.1002/col.21944",,,,
It is licenced under CC BY-NC (Attribution-NonCommercial):
https://creativecommons.org/licenses/by-nc/4.0/deed.en,,,,
Author: Dimitris Mylonas: dimitris.mylonas@nulondon.ac.uk,,,,
,,,,"""

import pandas as pd
import numpy as np
import seaborn as sns
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# process color names by removing whitespace and punctuation, and lowercasing all
# separate dataframe into input and output values
# return train and test sets afterwards
def preprocess(df):
    colors_cleaned = [c.strip().lower().translate(string.punctuation) for c in df['colour_name']]
    # remove whitespace and punctuation, lowercase all

    df.pop('colour_name') # remove the unprocessed color names, use processed ones
    df.pop('sample_id') # this column is unneeded
    df['colors'] = colors_cleaned # processed color names

    colors = df.pop('colors').values # y value, or the names of the colors
    rgb = df.values # x values, or the rgb values

    df['R'] = pd.to_numeric(df['R'])
    df['G'] = pd.to_numeric(df['G'])
    df['B'] = pd.to_numeric(df['B'])
    # to make sure all of the input values are numeric

    # setting up train/test
    X_train, X_test, y_train, y_test = train_test_split(rgb, colors, test_size=0.2, random_state=2)

    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    # need to normalize input values

    return X_train, X_test, y_train, y_test

def regress(upper_bound, X_train, X_test, y_train, y_test):
    # each element will be an array of:
    # [n_neighbors, precision_score, recall_score, f1_score]
    # takes upper bound of n_neighbors as input for organization
    fitness_scores = []
    # loop through n_neighbors to test each one's performance
    for i in range(1, upper_bound + 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        # KNeighborsClassifier for categorical output

        knn.fit(X_train, y_train) # fit the model
        y_pred = knn.predict(X_test) # test the model on test values

        prec = precision_score(y_pred, y_test, average='weighted', zero_division=0)
        rec = recall_score(y_pred, y_test, average='weighted', zero_division=0)
        f1 = f1_score(y_pred, y_test, average='weighted', zero_division=0)
        fitness_scores.append([i, prec, rec, f1])
        # get precision, recall, and f1 scores, add to array
        # zero_division=0 argument is used to avoid edge cases where output values
        # can be chosen 0 times; this is due to average=weighted argument

        print(f"the weighted precision score, weighted recall score, and weighted f1 score for {i} "
              f"neighbor(s) are {prec}, {rec}, and {f1} respectively")
        # print performance of each one

    return fitness_scores

# graph all of the fitness scores with n_neighbors in a multi line graph
def fitness_graphs(fitness_scores):
    graph_x_values = [f[0] for f in fitness_scores]
    graph_y1 = [f[1] for f in fitness_scores]
    graph_y2 = [f[2] for f in fitness_scores]
    graph_y3 = [f[3] for f in fitness_scores]

    plt.plot(graph_x_values, graph_y1, label ='weighted precision score')
    plt.plot(graph_x_values, graph_y2, label ='weighted recall score')
    plt.plot(graph_x_values, graph_y3, label ='weighted f1 score')
    # plot 3 different lines

    plt.xlabel("Number of neighbors in KN Classifier")
    plt.ylabel("Fitness score")
    plt.legend()
    plt.title('Different fitness scores by number of neighbors in KN Classifier')
    plt.show()

def main():
    df = pd.read_csv('colour_naming_data-1.csv')
    X_train, X_test, y_train, y_test = preprocess(df)
    fitness_scores = regress(12, X_train, X_test, y_train, y_test)
    fitness_graphs(fitness_scores)

if __name__ == '__main__':
    main()



