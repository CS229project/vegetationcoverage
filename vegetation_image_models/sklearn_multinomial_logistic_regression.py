import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    print(df.head())
    train_data = df.loc[df['year'].isin(train_set), :].to_numpy()[:,0:]
    eval_data = df.loc[df['year'].isin(eval_set), :].to_numpy()[:,0:]
    test_data = df.loc[df['year'].isin(test_set), :].to_numpy()[:,0:]
    print(train_data)

    #number_of_groups = np.unique(train_data[:,0]).shape[0]
    number_of_groups = int(np.max(train_data[:,1])) + 1
    print(number_of_groups)

    #train_y = get_one_hot_vector(number_of_groups, train_data[:, 0])
    #eval_y = get_one_hot_vector(number_of_groups, eval_data[:, 0])
    #test_y = get_one_hot_vector(number_of_groups, test_data[:, 0])

    train_y = train_data[:, :2]
    eval_y = eval_data[:, :2]
    test_y = test_data[:, :2]

    #Feature engineering
    #No feature engineering
    train_x = train_data[:, 2:]
    eval_x = eval_data[:, 2:]
    test_x = test_data[:, 2:]

    # Fit a Multimodal Regression model
    model = LogisticRegression(penalty='l2', multi_class='multinomial', random_state=0, max_iter=10000).fit(train_x, train_y[:,1])
    print(test_y[:,0].shape, model.predict(test_x).shape)
    
    y_predict = np.append(np.reshape(test_y[:,0], (test_y.shape[0],1)), np.reshape(model.predict(test_x),(test_y.shape[0],1)), 1)
    print(y_predict)
    np.savetxt('../data/k_' + str(number_of_groups) + '_prediction.txt', y_predict)

    '''
    train_losses, eval_losses = model.fit(train_x,train_y, eval_x, eval_y)
    
    plt.figure()
    iterations = np.arange(len(train_losses))
    plt.plot(iterations, train_losses, label="train")
    plt.plot(iterations, eval_losses, label="eval")
    plt.xlabel('iteration')
    plt.ylabel('losses')
    plt.legend()
    plt.savefig("loss_plot.png")

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    y_predict = model.predict(test_x)
    print(np.argmax(y_predict, axis=1, keepdims=True))
    print(np.unique(np.argmax(y_predict, axis=1, keepdims=True)))
    #Save Prediction
    predictions = np.argmax(y_predict, axis=1, keepdims=True)
    np.savetxt('../data/k_' + str(number_of_groups) + '_prediction.txt', predictions)
    '''

if __name__ == '__main__':
    main(lr=1,
        data_path='~/Downloads/k_12_data_no_encoding.csv',
        pred_path='../data/predictions.csv',
        start_year=2001,
        total_years=21)