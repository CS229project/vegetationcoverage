from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd


def init_centroids(centroid_file_name):
    """
    Initialize a `num_clusters` from a centroids value file

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters

    Returns
    -------
    centroids_init : nparray
        Centroids read from the parameter file
    """

    centroids_init = np.loadtxt(centroid_file_name)


    return centroids_init


def generate_image_from_data(image, centroids, prediction_file_name):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray
    prediction_file_name : string
        File with predictions with lables and position_encodings

    Returns
    -------
    image : nparray
        Updated image
    """

    pred_type = 1

    if pred_type == 1:
        prediction_file_name = '../data/k_4_data_predictions_sample.csv'
        df = pd.read_csv(prediction_file_name)
        print(centroids)
        
        #print(np.arcsin(df['pixel_position_encoding']))
        pixels_group = df['group'].to_numpy()
        pixels_group = np.reshape(pixels_group,(image.shape[0],image.shape[1],1))

        #total_pixels = image.shape[0]*image.shape[1]
        for i in range(pixels_group.shape[0]):
            for j in range(pixels_group.shape[1]):
                image[i,j] = centroids[pixels_group[i,j,0]]
                print(centroids[pixels_group[i,j,0]])
    elif pred_type == 0:
        prediction_file_name = '../data/k_4_prediction.txt'
        preds = np.loadtxt(prediction_file_name)

        #81790
        row = image.shape[0]
        col = image.shape[1]
        num_of_images = int(preds.shape[0]/(image.shape[0]*image.shape[1]))

        for k in range(num_of_images):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    print(preds[(i+1)*(j+1)])
                    print(centroids)
                    image[i,j] = centroids[int(preds[(k+1)*(i+1)*(j+1)-1])]
                    print(centroids[int(preds[(k+1)*(i+1)*(j+1)-1])])

    return image


def main(args):

    # Initialize image to be all black pixels
    image = np.zeros((254,322,3))

    # Setup
    prediction_file_name = args.data_path
    centroids_file_name = args.centroids_path
    image_name = "./prediction.png"
    figure_idx = 0

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids = init_centroids(centroids_file_name)
    num_clusters = centroids.shape[0]

    #Generate the image

 
    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Generating image ...')
    print(25 * '=')
    image_clustered = generate_image_from_data(image, centroids, prediction_file_name)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered/255)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'k_'+ str(num_clusters) + '_prediction.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')


    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/k_4_data_predictions_sample.csv',
                        help='Path to prediction file')
    parser.add_argument('--centroids_path', default='../data/k_4_centroids_rgb_values.dat',
                        help='Path to centroids file')
    args = parser.parse_args()
    main(args)
