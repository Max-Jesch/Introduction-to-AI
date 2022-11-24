import numpy as np
import matplotlib.pyplot as plt
import datetime, os
import matplotlib.image as mpimg
from IPython.display import Video, Image, HTML, display


def visualize_n_datapoints(pixels_array:np.array, labels:list, model=[] ,predicted_labels_desired=False):
    """
    visualize n datapoints by getting the predictions from the model and comparing them to the labels.
    The result is visualized in a table with the predicted labels and the actual labels

    !note: the index of the labels is incremented by 1 to match the index from the powerpoint slides
    
    Args:
        pixels_array (np.array): _description_
        labels (list): _description_
        model (list, optional): The current model. Defaults to [].
        predicted_labels_desired (boolean, optional): wether a prediction should be. Defaults to False.
    """
    green_check=mpimg.imread('images\\green_check.png')
    red_cross=mpimg.imread('images\\red_cross.png')
    
    plt.figure(1, figsize=(30, 30))
    for i in range(0,len(pixels_array)):
        pixels = pixels_array[i] 
        label = labels[i]
        plt.subplot(1,len(pixels_array)+1,i+1)
        plt.imshow(pixels.reshape(2,2, order='C'), cmap='gray_r')
        # add green check or red cross
        if predicted_labels_desired:
            predicted_label = model.predict(np.array([pixels]),1).argmax()
            if predicted_label == label:
                plt.imshow(green_check, extent=[0,1,1,0], alpha=0.1)  # extent is messed up :-/ https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
            else:
                plt.imshow(red_cross, extent=[0,1,1,0], alpha=0.1)    
            plt.title('correct Label :      {}\n predicted_label: {}'.format(label+1,predicted_label+1))
        else:
            plt.title('Label: {}'.format(label+1))
    plt.show()
    

def flatten_weights(model):
    """
    Flatten the weights of the model to a list of lists so that the weights can be visualized easily

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    weights=model.get_weights()
    flattened_weights=[]
    for weight_layer in weights:
        # ignore bias
        if len(weight_layer.shape)==2:
            # flatten the weights
            weight_layer=weight_layer.flatten()
            flattened_weights.append(weight_layer)
    return flattened_weights
            
            
def visualize_weights(model, image_path:str,save_image=False, result_image_name="undefined.png"):
    """
    Print the weights of the model over an image taken from the powerpoint slides

    Args:
        model (_type_): _description_
        image_path (str): _description_
    """
    flattened_weights=flatten_weights(model)
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots( figsize=(15, 15))
    ax.imshow(img)
    # those are the coordinates of all the "weights" in the image (extracted by hand :-/ )
    coordinates_list_list = [[(367,144), (352,213), (388,260),(420,300),(430,395),(393,447),(383,489),(424,530),(434,649),(406,687),(404,734),(386,774)],
                        [(622,99),(641,156),(672,211),(674,378),(630,465),(671,517),(662,643),(648,720),(633,812)],
                        [(914,179),(933,263),(961,317),(951,406),(932,459),(960,512),(960,607),(936,656),(936,723)]]
    
    for layer_index,coordinates_list in enumerate(coordinates_list_list):
        for weight_index, coordinates in enumerate(coordinates_list):
            # use color map to visualize the weights
            color = plt.cm.spring(flattened_weights[layer_index][weight_index]) # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            ax.text(coordinates[0],coordinates[1], str(flattened_weights[layer_index][weight_index])[:4], size=20, ha="center", va="center",
                    bbox=dict(boxstyle="round", color=color, ec="k", lw=1))
    if save_image:
        fig.savefig("images\\"+result_image_name)
    plt.show()
    
    
def create_datapoint()->tuple:
    """
    create a single datapoint with label
    NOTE: the index of the labels starts at 0 --> this is necessary for the one-hot encoding
    
    Returns:
        tuple: (np.array containing the "image", the label)
    """
    # create a random number between 0 and 1 
    pixel1=np.random.random()
    pixel2=np.random.random()
    pixel3=np.random.random()
    pixel4=np.random.random()
    
    puffer=0.4
    similarity=0.2
    
    if pixel1+pixel4-puffer>pixel2+pixel3 and abs(pixel1-pixel4)<similarity and abs(pixel2-pixel3)<similarity:
        label=0
        return np.array([pixel1,pixel2,pixel3,pixel4]),label
    elif pixel2+pixel3-puffer>pixel1+pixel4 and abs(pixel2-pixel3)<similarity and abs(pixel1-pixel4)<similarity: 
        label=1
        return np.array([pixel1,pixel2,pixel3,pixel4]),label
    elif pixel3+pixel4-puffer>pixel1+pixel2 and abs(pixel3-pixel4)<similarity and abs(pixel1-pixel2)<similarity:
        label=2
        return np.array([pixel1,pixel2,pixel3,pixel4]),label
    else:
        return(create_datapoint()) # try again if it does not fit any of the conditions


def create_dataset(size:int=1000)->tuple:
    """
    create a dataset with labels

    Args:
        size (int, optional): _description_. Defaults to 1000.

    Returns:
        tuple: _description_
    """
    examples=[]
    labels=[]
    for i in range(size):
        example,label=create_datapoint()
        examples.append(example)
        labels.append(label)
    return examples,labels

    

def show_two_images_side_by_side(image_path1,image_path2):
    """
    TODO: images are too small --> either zoom in or make them bigger on HTML page
    """

    # read images
    img_A = mpimg.imread(image_path1)
    img_B = mpimg.imread(image_path2)

    # display images
    # fig, ax = plt.subplots( figsize=(15, 15))

    fig, ax = plt.subplots(1,2,figsize=(30 ,20))
    ax[0].imshow(img_A)
    ax[1].imshow(img_B)
    ax[0].axis('off')
    ax[1].axis('off')