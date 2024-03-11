from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    #print(image.shape)
    ####REDO MAX
    #retarray = np.array([[]])
    #Pick any point as the first centroid
    #rand_index = np.random.randint(128, size=(2))
    #retarray = np.append(retarray, [image[rand_index[0], rand_index[1], :]], axis=1)
    #print(retarray)
    #Loop until you have filled up the list of the centroids
    #while retarray.shape[0] < num_clusters:
    #    #Look at how far each point is from the cluster centroids
    #    max_dist = 0
    #    max_dist_index = []
    #    for cluster_z in range(0, retarray.shape[0]):
    #        distances = np.linalg.norm(retarray[cluster_z, :] - image, axis=2, keepdims=True)
    #        max_index = np.unravel_index((np.argmax(distances),(image.shape[0], image.shape[1])))
            
    retarray = []
    index_1 = np.random.randint(image.shape[0], size=(num_clusters))
    index_2 = np.random.randint(image.shape[1], size=(num_clusters))
    for i in range(0, num_clusters):
        #Pick any point at random
        #for index in indices:
        retarray.append(image[index_1[i], index_2[i],:])
    
    centroids_init = np.array(retarray)
    print(centroids_init)
    #raise NotImplementedError('init_centroids function not implemented')

    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    #for the max number of iterations
    i = 0
    H,W,C = image.shape

    #max_iter = 1
    while i < max_iter:
        #Calculate the distance between the points to each centroid
        #Make the one with the least ditance the centroid
        distances_stack = []
        for centroid in centroids:
            distances = np.linalg.norm(centroid - image, axis=2, keepdims=True)
            distances_stack.append(distances)

        i += 1
        pixel_distances = np.stack(distances_stack, axis=2)   #Gives   1424, 1661, 16, 1
        pixel_distances = np.squeeze(pixel_distances, axis=3) #Makes   1424, 1661, 16
        #print((pixel_distances.sum(axis=2)))
        pixels_group = np.argmin(pixel_distances, axis=2, keepdims=True) #Gives 128, 128, 1 (group)

        #Recalculate the centroids
        #Get all the pixels in the same group
        new_centroids = []
        for group_z in range(centroids.shape[0]):
            group_z_indices = np.transpose((pixels_group==group_z).nonzero())

            group_pixels = []
            for group_z_index in group_z_indices:
                #Collate all the pixel images
                group_pixels.append(image[group_z_index[0],group_z_index[1],:])
        
            group_pixels = np.array(group_pixels)
            print(group_pixels.shape)
            new_centroids.append(np.mean(group_pixels,axis=0))
        centroids = np.array(new_centroids)
         
    new_centroids = centroids
    print('Iterations Completed:', i)
    #raise NotImplementedError('update_centroids function not implemented')
    # *** END YOUR CODE ***

    return new_centroids

def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    distances_stack = []
    print(centroids)
    for centroid in centroids:
        distances = np.linalg.norm(centroid - image, axis=2, keepdims=True)
        distances_stack.append(distances)

    pixel_distances = np.stack(distances_stack, axis=2)   #Gives   512, 512, 16, 1
    pixel_distances = np.squeeze(pixel_distances, axis=3) #Makes   512, 512, 16
    pixels_group = np.argmin(pixel_distances, axis=2, keepdims=True) #Gives 512, 512, 1 (group)
    
    for i in range(pixels_group.shape[0]):
        for j in range(pixels_group.shape[1]):
            image[i,j] = centroids[pixels_group[i,j,0]]

    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.data_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    scatterData = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
    print(scatterData.shape)
    
    plt.figure(figure_idx)
    figure_idx += 1
    plt.title('Image Scatter plot')
    plt.axis('off')
    plt.scatter(scatterData[:,0],scatterData[:,1],scatterData[:,2], linewidths=1, alpha=.7,
           edgecolor='k',
           c=scatterData[:,2])
    savepath = os.path.join('.', 'scatter.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    
    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')


    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../images/landsat_manaus_2001.tif',
                        help='Path to small image')
    parser.add_argument('--large_path', default='../images/landsat_manaus_2001.tif',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=8,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
