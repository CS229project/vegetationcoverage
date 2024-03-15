import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

def get_one_hot_vector(number_of_groups, y):
    #print(y.shape)
    #print(number_of_groups)
    one_hot_y = np.zeros((y.shape[0], number_of_groups+1))
    one_hot_y[np.arange(y.shape[0]).astype(int), y.astype(int)] = 1
    
    return one_hot_y

def main(lr, data_path, pred_path, start_year, total_years):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training, test and validation set
    total_set = np.arange(0,total_years) + start_year
    train_split = 0.8
    eval_split = 0.1

    train_set=np.random.choice(total_set, size=int(total_years*train_split), replace=False)
    eval_test_set = np.setdiff1d(total_set, train_set)
    eval_set = np.random.choice(eval_test_set, size=int(total_years*eval_split), replace=False)
    test_set = np.setdiff1d(eval_test_set, eval_set)

    #Load data from the csv
    df = pd.read_csv(data_path,index_col=False)
    train_data = df.loc[df['year'].isin(train_set), :].to_numpy()[:,0:]
    eval_data = df.loc[df['year'].isin(eval_set), :].to_numpy()[:,0:]
    test_data = df.loc[df['year'].isin(test_set), :].to_numpy()[:,0:]

    number_of_groups = int(np.max(train_data[:,1])) + 1

    train_y = train_data[:, :2]
    eval_y = eval_data[:, :2]
    test_y = test_data[:, :2]

    #Feature engineering
    #No feature engineering

    print(int(np.max(train_data[:, 2])))
    print(int(np.max(train_data[:, 3])))
    train_x = train_data[:, 2:]
    print(train_x)
    one_train_x = get_one_hot_vector(int(np.max(train_data[:, 2])),train_data[:, 2].astype(int))
    one_train_y = get_one_hot_vector(int(np.max(train_data[:, 3])),train_data[:, 3].astype(int))
    one_hot_image = np.append(one_train_x, one_train_y,1)
    train_x = np.append(one_hot_image, train_x[:, 2:],1)

    eval_x = eval_data[:, 2:]
    one_eval_x = get_one_hot_vector(int(np.max(eval_data[:, 2])),eval_data[:, 2].astype(int))
    one_eval_y = get_one_hot_vector(int(np.max(eval_data[:, 3])),eval_data[:, 3].astype(int))
    one_hot_image = np.append(one_eval_x, one_eval_y,1)
    eval_x = np.append(one_hot_image, eval_x[:, 2:],1)

    test_x = test_data[:, 2:]
    one_test_x = get_one_hot_vector(int(np.max(test_data[:, 2])),test_data[:, 2].astype(int))
    one_test_y = get_one_hot_vector(int(np.max(test_data[:, 3])),test_data[:, 3].astype(int))
    one_hot_image = np.append(one_test_x, one_test_y,1)
    test_x = np.append(one_hot_image, test_x[:, 2:],1)

    print(train_x.shape)
    print(eval_x.shape)
    print(test_x.shape)


    #1 add position sine value to data
    #train_x = train_data[:, 3:] + np.reshape(train_data[:, 2], (train_data.shape[0],1)
    #eval_x = eval_data[:, 3:] + np.reshape(eval_data[:, 2] ,(eval_data.shape[0],1))
    #test_x = test_data[:, 3:] + np.reshape(test_data[:, 2],(test_data.shape[0],1))



    # Fit a Multimodal Regression model
    model = LogisticRegression(penalty='l2', multi_class='multinomial', class_weight='balanced', random_state=0, max_iter=10, n_jobs=-2, verbose=1).fit(train_x, train_y[:,1])
    
    y_predict = np.append(np.reshape(test_y[:,0], (test_y.shape[0],1)), np.reshape(model.predict(test_x),(test_y.shape[0],1)), 1)
    np.savetxt('../data/k_' + str(number_of_groups) + '_prediction.txt', y_predict)

    cm = confusion_matrix(test_y[:,1], y_predict[:,1])
    print(f'tn: {cm[0, 0]}, fp: {cm[0, 1]}, fn: {cm[1, 0]}, tp: {cm[1, 1]}')
    print(f'Accuracy Score: {accuracy_score(test_y[:,1], y_predict[:,1])}')

 
if __name__ == '__main__':
    main(lr=1,
        data_path='~/Downloads/k_4_data_pixel_x_y.csv',
        pred_path='../data/predictions.csv',
        start_year=2001,
        total_years=21)