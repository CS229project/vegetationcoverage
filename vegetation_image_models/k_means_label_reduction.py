from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
from skimage import io, transform


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

    retarray = []
    index_1 = np.random.randint(image.shape[0], size=(num_clusters))
    index_2 = np.random.randint(image.shape[1], size=(num_clusters))
    for i in range(0, num_clusters):
        #Pick any point at random
        #for index in indices:
        retarray.append(image[index_1[i], index_2[i],:])
    
    centroids_init = np.array(retarray)
    print(centroids_init)


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

    #for the max number of iterations
    i = 0
    H,W,C = image.shape
    prev_distance = 0

    #max_iter = 1
    while i < max_iter:
        #Calculate the distance between the points to each centroid
        #Make the one with the least ditance the centroid
        distances_stack = []
        for centroid in centroids:
            distances = np.linalg.norm(centroid - image, axis=2, keepdims=True)
            distances_stack.append(distances)

        i += 1
        pixel_distances = np.stack(distances_stack, axis=2)                  #Gives   844, 1074, 16, 1
        pixel_distances = np.squeeze(pixel_distances, axis=3)                #Makes   844, 1075, 16
        pixel_distances_min = np.min(pixel_distances, axis=2,keepdims=True)  #Take the lowest distance for each pixel

        if prev_distance != pixel_distances_min.sum():
            prev_distance = pixel_distances_min.sum()
        else:
            break

        pixels_group = np.argmin(pixel_distances, axis=2, keepdims=True) #Gives 1424, 1661, 1 (group)

        #Recalculate the centroids
        #Get all the pixels in the same group
        new_centroids = []
        for group_z in range(centroids.shape[0]):
            group_z_indices = np.transpose((pixels_group==group_z).nonzero())

            group_pixels = []
            for group_z_index in group_z_indices:
                #Collate all the pixel images
                group_pixels.append(image[group_z_index[0],group_z_index[1],:])
        
            if (len(group_pixels) == 0):
                group_pixels = [[0,0,0]]

            group_pixels = np.array(group_pixels)
            new_centroids.append(np.mean(group_pixels,axis=0))
        centroids = np.array(new_centroids)
         
        if ((i % print_every) == 0): print(f'Iteration {i} Heterogeneity Measure: {prev_distance}')

    new_centroids = centroids
    print(f'Iterations Completed:{i} Heterogeneity Measure: {prev_distance}')

    return new_centroids


def get_one_hot_vector(number_of_groups, y):
    #print(y.shape)
    #print(number_of_groups)
    one_hot_y = np.zeros((y.shape[0], number_of_groups+1))
    one_hot_y[np.arange(y.shape[0]).astype(int), y.astype(int)] = 1
    return one_hot_y


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P


def generate_image_and_data(image, centroids, image_name, year, year_csv_value, f):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray
    image_name : string
        The name of the image to store the data
    year: string
        The year for which the data is being labeled
    year_csv_value: string
        Values for emissions for that particular year
    f: filehandler
        Filehandler to write the data

    Returns
    -------
    image : nparray
        Updated image
    """

    distances_stack = []
    for centroid in centroids:
        distances = np.linalg.norm(centroid - image, axis=2, keepdims=True)
        distances_stack.append(distances)

    pixel_distances = np.stack(distances_stack, axis=2)              #Gives   844, 1074, 16, 1
    pixel_distances = np.squeeze(pixel_distances, axis=3)            #Makes   844, 1074, 16
    pixels_group = np.argmin(pixel_distances, axis=2, keepdims=True) #Gives   844, 1074, 1 (group)


    
    total_pixels = image.shape[0]*image.shape[1]
    pixel_position_encodings =  getPositionEncoding(total_pixels, d=128, n=10000)
    print(pixel_position_encodings.shape)
    for i in range(pixels_group.shape[0]):
        for j in range(pixels_group.shape[1]):
            image[i,j] = centroids[pixels_group[i,j,0]]
            
            #Write the labels and the positional data
            pixel_position = (i+1)*(j+1)
            #encoding = np.sin(pixel_position*np.pi*((2*total_pixels)**-1)) #Positional encoding from 0 to 1 using sin
            #encoding = pixel_position #Positional encoding from 0 to 1 using sin
            #encoding = f'{str(i+1)},{str(j+1)}'
            
            ####Cosine Encoding
            encoding = ','.join(str(encoding) for encoding in pixel_position_encodings[pixel_position-1, :])

            ####One Hot encoding
            #one_hot = get_one_hot_vector(total_pixels, np.array([pixel_position-1]))
            #encoding = ','.join(str(val) for val in one_hot[0])
            #print(encoding + '\n')

            f.write(f'{year},{str(pixels_group[i,j,0])},{str(encoding)},{year_csv_value}\n')
            #print(f'{year},{str(pixels_group[i,j,0])},{str(encoding)},{year_csv_value}\n')
    return image


def main(args):

    data_gen_from_scratch = True
    compression_factor = 1/3 

    if data_gen_from_scratch:

        #Open data for carbon dioxide, methane, ntrs_oxide and srfce_tmp
        #We have data available till 2021
        df_cs = pd.read_csv('../emmissions_data/clean_data/crbn_dioxide_complete.csv', header=None).to_numpy()
        df_mt = pd.read_csv('../emmissions_data/clean_data/methane_complete.csv', header=None).to_numpy()
        df_ntrs = pd.read_csv('../emmissions_data/clean_data/ntrs_oxide_complete.csv', header=None).to_numpy()
        df_st = pd.read_csv('../emmissions_data/clean_data/srfce_tmp_afft_complete.csv', header=None).to_numpy()


        # Setup
        max_iter = args.max_iter
        print_every = args.print_every
        image_path_small = args.data_path
        image_path_large = args.large_path
        num_clusters = args.num_clusters
        figure_idx = 0
        image_name = image_path_large.split('/')[-1].split('.')[0]
        base_year = image_path_small.split('_')[-1].split('.')[0]


        #Normalize the values
        df_cs[:,1] = df_cs[:,1] / df_cs[:,1].max(axis=0)
        #Convert iteger to float
        df_mt = df_mt.astype(float)
        df_mt[:,1] = df_mt[:,1] / df_mt[:,1].max(axis=0)
        df_ntrs[:,1] = df_ntrs[:,1] / df_ntrs[:,1].max(axis=0)
        df_st[:,1] = df_st[:,1] / df_st[:,1].max(axis=0)

        # Load small image
        image = np.copy(mpimg.imread(image_path_small))
        image = transform.rescale(image, scale=(compression_factor,compression_factor,1))
        print(image.shape)
        print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
        plt.figure(figure_idx)
        figure_idx += 1
        plt.imshow(image)
        plt.title('Original small image')
        plt.axis('off')
        savepath = os.path.join('.', 'orig_small.png')
        plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')


        #Scatter plot the image to check if we can identify clusters
        scatterData = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
        plt.figure(figure_idx)
        figure_idx += 1
        plt.title('Image Scatter plot')
        plt.axis('off')
        plt.axes(projection = '3d', proj_type = 'ortho')
        plt.xlabel('Red')
        plt.ylabel('Green')

        plt.scatter(scatterData[:,0]/255,scatterData[:,1]/255,scatterData[:,2]/255, c=scatterData/255)
        savepath = os.path.join('.', 'scatter.png')
        plt.savefig(savepath, transparent=False, format='png', bbox_inches='tight')

        # Initialize centroids
        print('[INFO] Centroids initialized')
        centroids_init = init_centroids(num_clusters, image)

        # Update centroids
        print(25 * '=')
        print('Updating centroids ...')
        print(25 * '=')
        centroids = update_centroids(centroids_init, image, max_iter, print_every)

        # Load large image
        #image = np.copy(mpimg.imread(image_path_large))
        image = np.copy(mpimg.imread(image_path_large))
        image = transform.rescale(image, scale=(compression_factor,compression_factor,1))
        total_pixels = image.shape[0]*image.shape[1]
        image.setflags(write=1)
        print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
        plt.figure(figure_idx)
        figure_idx += 1
        plt.imshow(image)
        plt.title('Original large image')
        plt.axis('off')
        savepath = os.path.join('.', 'orig_large.png')
        plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')



        #Start writing the data as well
        f = open("../data/k_" + str(centroids.shape[0]) + "_Mar19_data_cosine_" + str(image.shape[0]) + "_" + str(image.shape[1]) + ".csv", "w")
        #f.write("year,group,pixel_position_encoding_x,pixel_position_encoding_y,crbn_dioxide,methane,ntrs_oxide,srfce_tmp\n")
        #f.write("year,group,pixel_position_encoding,crbn_dioxide,methane,ntrs_oxide,srfce_tmp\n")
        columns = ''
        total_pixels = image.shape[0]*image.shape[1]
        for i in range(total_pixels):
            columns = columns + 'pixeloh_' + str(i) + ','
        header = f'year,group,{columns}crbn_dioxide,methane,ntrs_oxide,srfce_tmp\n'
        #f.write("year,group,pixel_position_encoding_0,pixel_position_encoding_1,pixel_position_encoding_2,pixel_position_encoding_3,pixel_position_encoding_4,pixel_position_encoding_5,pixel_position_encoding_6,pixel_position_encoding_7,pixel_position_encoding_8,pixel_position_encoding_9,pixel_position_encoding_10,pixel_position_encoding_11,pixel_position_encoding_12,pixel_position_encoding_13,pixel_position_encoding_14,pixel_position_encoding_15,crbn_dioxide,methane,ntrs_oxide,srfce_tmp\n")
        f.write(header)

        #Run through all the images and generate the dataset
        basepath = '../images/'
        with os.scandir(basepath) as entries:
            for entry in entries:
                if entry.is_file():
                    if (entry.name != ".DS_Store"):
                        year = entry.name.split('.')[0].split('_')[-1]
                        image_name = basepath + entry.name

                        year_csv_value = str(df_cs[(df_cs[:,0] == int(year)).nonzero()][0,1])
                        year_csv_value = year_csv_value + ',' + str(df_mt[(df_mt[:,0] == int(year)).nonzero()][0,1])
                        year_csv_value = year_csv_value + ',' + str(df_ntrs[(df_ntrs[:,0] == int(year)).nonzero()][0,1])
                        year_csv_value = year_csv_value + ',' + str(df_st[(df_st[:,0] == int(year)).nonzero()][0,1])

                        # Load next image image
                        print(image_name)
                        image = np.copy(mpimg.imread(image_name))
                        image = transform.rescale(image, scale=(compression_factor,compression_factor,1))
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
                        print('Updating large image ...' + image_name)
                        print(25 * '=')
                        image_clustered = generate_image_and_data(image, centroids, image_name, year, year_csv_value, f)

                        plt.figure(figure_idx)
                        figure_idx += 1
                        plt.imshow(image_clustered)
                        plt.title('Updated large image')
                        plt.axis('off')
                        savepath = os.path.join('.', 'k_'+ str(num_clusters) + '_' + year + '_updated_large_' + base_year + '_base.png')
                        plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

        f.close()
        #Save the centroids
        np.savetxt("../data/k_" + str(centroids.shape[0]) + "_centroids_rgb_values_Mar19_" + str(compression_factor**-1) + ".dat", centroids)
    else:
        df_pred = pd.read_csv('../emmissions_data/predicted_ghg_values.csv').to_numpy()
        df_images = pd.read_csv('~/Downloads/k_4_data_Mar12.csv')

        #Start writing the data as well
        f = open("../data/k_4_prediction_Mar12.txt", "w")
        f.write("year,group,pixel_position_encoding_0,pixel_position_encoding_1,pixel_position_encoding_2,pixel_position_encoding_3,pixel_position_encoding_4,pixel_position_encoding_5,pixel_position_encoding_6,pixel_position_encoding_7,crbn_dioxide,methane,ntrs_oxide,srfce_tmp\n")

        finalData = None
        #loop through the years
        for i in range(df_pred.shape[0]):
            print(df_images[df_images['year']==df_pred[i,0]])
            df_curr = df_images[df_images['year']==df_pred[i,0]].to_numpy()
            data = df_curr[:,0:3]
            data = np.append(data, np.tile(df_pred[i,1:], (data.shape[0],1)),1)
            #data.tofile("../data/k_4_" + str(int(df_pred[i,0])) +  "_predictions_data.csv", sep = ',')
            for i in range(data.shape[0]):
                line = f'{data[i,0]},{int(data[i,1])},{data[i,2]},{data[i,3]},{data[i,4]},{data[i,5]},{data[i,6]}\n'
                f.write(line)

        f.close()



    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../images/NDVI_landsat_manaus_2001.tif',
                        help='Path to base image')
    parser.add_argument('--large_path', default='../images/NDVI_landsat_manaus_2021.tif',
                        help='Path to later image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=4,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
