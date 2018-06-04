import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def return_cumulative_data():
    df = pd.read_csv('TrainingSet_clean.csv', encoding='latin-1')

    rowID = df.pop('row ID').values
    df = pd.get_dummies(df)
    Y = df.pop('case_status').values
    X = df.values

    clf = RandomForestClassifier()
    clf = clf.fit(X, Y)

    feature_importances = pd.DataFrame(clf.feature_importances_,
                                       index=df.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    feature_importances_sum = feature_importances.cumsum()

    index_to_get = feature_importances_sum[feature_importances_sum.values <= 0.95].index.values

    return index_to_get


def binarised_data(filename, is_training):
    df = pd.read_csv(filename)

    df = pd.get_dummies(df)

    index_to_get = return_cumulative_data()

    if(is_training):
        index_to_get = np.append(index_to_get, 'case_status')

    ## IF TRAINING, ADD CASE_STATUS
    df = df[index_to_get]
    df.sort_index(axis=1, inplace=True)

    return df

binarised_data('TrainingSet_clean.csv', is_training=True).to_csv('TrainingSet_binarised.csv', index=False)
binarised_data('TestingSet_clean.csv', is_training=False).to_csv('TestingSet_binarised.csv', index=False)