"""Comments on csv:
This colour naming dataset was collected @colornaming.net and published in Mylonas, D., &
MacDonald, L. (2016). Augmenting basic colour terms in English. Color Research & Application,
41(1), 32�42. https://doi.org/10.1002/col.21944",,,,
It is licenced under CC BY-NC (Attribution-NonCommercial):
https://creativecommons.org/licenses/by-nc/4.0/deed.en,,,,
Author: Dimitris Mylonas: dimitris.mylonas@nulondon.ac.uk,,,,
,,,,"""

import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('colour_naming_data-1.csv')

colors_cleaned = [c.strip().lower().translate(string.punctuation) for c in df['colour_name']]
# remove whitespace, lowercase, remove punctuation

le = LabelEncoder()
le.fit(colors_cleaned)
color_numbers = le.transform(colors_cleaned)
print(color_numbers)

df.pop('colour_name')
df.pop('sample_id') # unneeded
df['colors'] = color_numbers
df['colors'] = pd.to_numeric(df['colors'])
# remove raw color names, add cleaned ones

colors = df.pop('colors').values # y value, or the names of the colors
rgb = df.values # x values, or the rgb values

df['R'] = pd.to_numeric(df['R'])
df['G'] = pd.to_numeric(df['G'])
df['B'] = pd.to_numeric(df['B'])


# setting up train/test
X_train, X_test, y_train, y_test = train_test_split(rgb, colors, test_size=0.2, random_state=1)
print(f"{len(X_test)} {len(y_test)}")

for i in range(1, 12):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)

    x_pred = [knn.predict([x]) for x in X_test]
    # accuracy = knn.score(X_test, y_test)

    counter = 0
    for i in range(len(x_pred)):
        x, y = x_pred[i], y_test[i]
        if x == y:
            counter += 1

    print(float(counter / 1211))



