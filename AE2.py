"""Comments on csv:
This colour naming dataset was collected @colornaming.net and published in Mylonas, D., &
MacDonald, L. (2016). Augmenting basic colour terms in English. Color Research & Application,
41(1), 32�42. https://doi.org/10.1002/col.21944",,,,
It is licenced under CC BY-NC (Attribution-NonCommercial):
https://creativecommons.org/licenses/by-nc/4.0/deed.en,,,,
Author: Dimitris Mylonas: dimitris.mylonas@nulondon.ac.uk,,,,
,,,,"""

"""
TODO
most rgb values have duplicates, for each rgb value find the most common prediction 
for it, and use that value in the dataset; the dataset should only have one of
each rgb value - DONE

for the centroids of the colors (part 2) (use original dataset for this):
for each color, average out every set of rgb values that was predicted to be 
that color, and treat the resulting mean as the centroid
then for each color that is predicted, take the euclidean distance between
the input rgb, and the centroid rgb of that color
"""

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
from statistics import mode

def unique_rgb_parsing(df):
    # finding most common color prediction per rgb value
    df = pd.read_csv('colour_naming_data-1.csv')
    colors_cleaned = [c.strip().lower().translate(string.punctuation) for c in df['colour_name']]
    # remove whitespace, punctuation, and lowercase the whole color name

    df.pop('colour_name') # remove the unprocessed color names, use processed ones
    df.pop('sample_id') # this column is unneeded
    df['colors'] = colors_cleaned # processed color names

    rgb_to_color = {} # dict used to store color predictions per rgb
    unique_rgb = [] # 2d list that i will create new dataframe from

    for index, row in df.iterrows():
        rgb = tuple([row['R'], row['G'], row['B']]) # will be used as key
        if rgb in rgb_to_color.keys():
            rgb_to_color[rgb].append(row['colors'])
            # each rgb will correspond to a list of all the predictions it has
            # if this rgb value has been seen before, append current prediction to value list
        else:
            rgb_to_color[rgb] = [row['colors']]
            # if this rgb value is new, create new list as its value with current prediction

    for rgb in rgb_to_color.keys():
        colors = rgb_to_color[rgb]
        most_common = mode(colors) # statistics.mode gets most common prediction
        # in the case of a tie, the first one to appear will be used
        unique_rgb.append([int(rgb[0]), int(rgb[1]), int(rgb[2]), most_common])
        # 2d list, each inner list is [R, G, B, color name]

    df2 = pd.DataFrame(unique_rgb, columns=['R', 'G', 'B', 'colors'])
    # create & return new dataframe from unique_rgb 2d list
    return df2

# separate dataframe into input and output values
# return train and test sets afterwards
def preprocess(df):
    colors = df.pop('colors').values  # y value, or the names of the colors
    rgb = df.values  # x values, or the rgb values

    # setting up train/test
    X_train, X_test, y_train, y_test = train_test_split(rgb, colors,
                                            test_size=0.2, random_state=4)

    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    # need to normalize input values

    return X_train, X_test, y_train, y_test

def regress(upper_bound, X_train, X_test, y_train, y_test):
    # each element of fitness_scores will be an array of:
    # [n_neighbors, precision_score, recall_score, f1_score]
    # this function takes upper bound of n_neighbors as input
    fitness_scores = []
    # loop through n_neighbors to test each one's performance
    for i in range(1, upper_bound + 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        # KNeighborsClassifier for categorical output

        knn.fit(X_train, y_train) # fit the model
        y_pred = knn.predict(X_test) # test the model on test values

        prec = round(precision_score(y_pred, y_test, average='weighted',
                                     zero_division=0), 3)
        rec = round(recall_score(y_pred, y_test, average='weighted',
                                 zero_division=0), 3)
        # the recall score is equal to the knn.score metric
        f1 = round(f1_score(y_pred, y_test, average='weighted', zero_division=0), 3)
        # get precision, recall, and f1 scores, add to array
        # zero_division=0 argument is used to avoid edge cases where output values
        # can be chosen 0 times; this is due to average=weighted argument
        fitness_scores.append([i, prec, rec, f1])

        print(f"The weighted precision score, weighted recall score, "
              f"and weighted f1 score for {i} "
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
    df2 = unique_rgb_parsing(df) # df2 will be the dataframe with unique values
    X_train, X_test, y_train, y_test = preprocess(df2) # returns train/test values
    fitness_scores = regress(12, X_train, X_test, y_train, y_test)
    # generates fitness scores with the train/test values
    fitness_graphs(fitness_scores) # generates graphs from fitness_scores

if __name__ == '__main__':
    main()



