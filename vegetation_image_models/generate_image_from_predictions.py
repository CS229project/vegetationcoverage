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

    pred_type = 0

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
        preds = pd.read_csv(prediction_file_name)

        years = np.unique(preds['year'].to_numpy())
        num_of_images = years.shape[0]

        for k in range(num_of_images):
            print(f'Generating image {years[k]}')
            pixels_group = preds[preds['year']==years[k]]
            print(pixels_group)
            pixels_group = pixels_group['group'].to_numpy()
            print(pixels_group.shape)
            pixels_group = np.reshape(pixels_group,(image.shape[0],image.shape[1]))
            print(pixels_group.shape)

            #curr_pred = preds[preds[:,0] == years[k]]
            for i in range(pixels_group.shape[0]):
                for j in range(pixels_group.shape[1]):
                    #print(pixels_group[i,j])
                    image[i,j] = centroids[int(pixels_group[i,j])]
            plt.figure(k)
            plt.imshow(image)
            plt.title('Updated large image')
            plt.axis('off')
            savepath = os.path.join('.', 'k_'+ str(int(years[k])) + '_prediction.png')
            plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')
            image = np.zeros((image.shape[0],image.shape[1],image.shape[2]))



    return image


def main(args):

    # Initialize image to be all black pixels
    #image = np.zeros((254,322,3))
    #image = np.zeros((51,64,3))
    image = np.zeros((85,107,3))


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
    parser.add_argument('--data_path', default='../data/k_4_prediction_mar18.txt',
                        help='Path to prediction file')
    parser.add_argument('--centroids_path', default='../data/k_4_centroids_rgb_values_Mar18_3.0.dat',
                        help='Path to centroids file')
    args = parser.parse_args()
    main(args)
