#-------------------------------------------#
#        Written by Mahsa Paknezhad         #
#                 Feb 2024                  #
#-------------------------------------------#

import numpy as np
import torch
import os
import pandas as pd
import clip
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import psutil
import time



def load_data(data_path, csv_file, preprocess):
    """
    Load the images and their corresponding captions.

    Parameters:
    data_path (str): Location of the images and the CSV file containing the image-caption pairs.
    csv_file (str): Name of the CSV file containing the image-caption pairs. 
    preprocess (torchvision.transforms.Compose): A set of transformations to apply to the input images.

    Returns:
    original_images (List): The images listed in the input CSV file and whose corresponding files are available in the 'data_path'
    captions (List): The captions for the above images
    images(List): The images in the 'original_images' list after going through the input transformations
    urls (List): The url of the above images
    """

    original_images = []
    images = []
    captions = []
    urls = []

    # Read the csv file containing the image-caption pairs 
    pairs = pd.read_csv(os.path.join(data_path, csv_file), index_col=0)

    # For each row in the csv file
    for  row in pairs.iterrows():
        # Get the image filename
        filename = row[0]
        name = filename.split('/')[4]+'.png'
        
        # Get the corresponding caption
        caption = row[1]['caption']
        # Plot the image if the image exists in the data folder
        if os.path.isfile(os.path.join(data_path, name)):
            image = Image.open(os.path.join(data_path, name)).convert("RGB")

            # Keep the origin image and caption
            original_images.append(image)
            captions.append(caption)
            # Apply the image preprocessing pipeline to the image
            images.append(preprocess(image))
            urls.append(filename)

    return images, captions, original_images, urls

def plot_outputs(data_path, images, captions, sim_matrix):
    """
    Plot the images, their corresponding captions and the measured similarity for each image-caption pair.

    Parameters:
    data_path (str): Location of the images and the CSV file containing the image-caption pairs.
    images(List): The images in the input CSV file after going through the input transformations
    captions (List): The captions for the above images
    sim_matrix (numpy.ndarray): A square matrix showing cosine similarity between image-caption pairs
    """

    # Define a figure to plot the image-caption pairs
    fig = plt.figure(figsize=(10, 60))
    # Counter for subplots
    k = 1
    # For each row in the csv file
    for  n, (image, caption) in enumerate(zip(images, captions)):
        
            # Show the image
            plt.subplot(len(images)//2+1, 4, k)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            # Show the caption
            plt.subplot(len(images)//2+1, 4, k+1)
            caption = caption.replace('\n', ' ')

            # Break caption into smaller pieces
            title = ''
            tx_len = 50
            if len(caption)//tx_len == 0:
                title = caption
            else:
                for i in range(len(caption)//tx_len):
                    title += caption[i*tx_len: (i+1)*tx_len]+'\n'
                title += caption[(i+1)*tx_len:]
            plt.text(.01, .80,f"{title}", fontsize=6, va='top')
            plt.text(.01, .90, "Cosine similarity: %0.02f"%(sim_matrix[n,n]), color='red', ha='left', va='top')
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.tight_layout()

            # Update the counter for subplots
            k+=2
    
    fig.savefig(os.path.join(data_path, 'image_captions.jpg'), dpi=300, bbox_inches='tight')
    return 

def write_outputs(data_path, csv_file, urls, sim_matrix):
    """
    Writes the measured cosine similarity for each image and its corresponding caption as a new column in the input CSV file.

    Parameters:
    data_path (str): Location of the images and the CSV file containing the image-caption pairs.
    csv_file (str): Name of the CSV file containing the image-caption pairs. 
    urls (List): The url of the images in the CSV file whose corresponding files are available in the 'data_path'
    sim_matrix (numpy.ndarray): A square matrix showing cosine similarity between image-caption pairs
    """

    # Get the similarity values for the image-captions of interest
    similarity = np.diagonal(sim_matrix)
    # Read the csv file containing the image-caption pairs 
    pairs = pd.read_csv(os.path.join(data_path, csv_file), index_col=0)
    similarity_dict = {'url': urls, 'cosine_sim': similarity}
    similarity_df = pd.DataFrame(similarity_dict)
    new_df = pd.merge(pairs, similarity_df, how="left", on=["url"])
    new_df.to_csv(os.path.join(data_path, 'output.csv'))
    return

def plot_matrix(data_path, sim_matrix):
    """
    Plots the measured cosine similarity for each image and every caption in the input CSV file.

    Parameters:
    data_path (str): Location of the images and the CSV file containing the image-caption pairs.
    sim_matrix (numpy.ndarray): A square matrix showing cosine similarity between image-caption pairs
    """
    # Plot the similarity matrix
    plt.figure(figsize=(20, 14))
    plt.imshow(sim_matrix, vmin=0.1, vmax=0.41)

    plt.yticks([])
    plt.xticks([])
    # Write the similarity value in text format on each matrix cell
    for x in range(sim_matrix.shape[1]):
        for y in range(sim_matrix.shape[0]):
            plt.text(x, y, f"{sim_matrix[y, x]:.2f}", ha="center", va="center", size=5)

    # Set labels and range of values for the figure
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, sim_matrix.shape[0] - 0.5])
    plt.ylim([sim_matrix.shape[0] + 0.5, -2])
    plt.xlabel('Captions', fontsize=20)
    plt.ylabel('Images', fontsize=20)
    
    # Set the title and save the figure
    plt.title("Cosine similarity between text and image features", size=20)
    plt.savefig(os.path.join(data_path, 'similarity_matrix.jpg'), dpi=300)
    return

if __name__=="__main__":
    # Start memory profiling
    process = psutil.Process()
    # Start timing
    start_time = time.time()
    # Define input parameters
    parser = argparse.ArgumentParser(description="Measuring Cosine Similarity between Images and Captions")
    parser.add_argument("--data_path", type=str, default='/home/ubuntu/projects/CLIP/data/challenge_set/',
                        help='path to the dataset directory')
    parser.add_argument("--csv_file", type=str, default='challenge_set.csv',
                        help='name of the csv file in data_path that contains the image names and their'\
                        'corresponding captions')
    args = parser.parse_args()

    # Load the ViT model and its image preprocessing pipeline
    model, preprocess = clip.load("ViT-B/32")
    # Put the model on GPU and set it to eval mode
    model.cuda().eval()

    # Get the image and caption pairs and show them in a figure
    imgs, cptns, orig_imgs, urls = load_data(args.data_path, args.csv_file, preprocess)

    # Convert the batch of images into a tensor and move it to GPU 
    image_input = torch.tensor(np.stack(imgs)).cuda()
    # Tokenize the captions and move them to GPU
    text_tokens = clip.tokenize(["This is " + cptn for cptn in cptns], context_length=77, truncate=True).cuda()

    # Get image and caption features from CLIP
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    # Measure cosine similarity of image features to text features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    # Write the similarity values as an additiona column in the input csv file
    write_outputs(args.data_path, args.csv_file, urls, similarity)
    
    # Finish timing
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time, "seconds")
    # Finish memory profiling
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    print("Memory usage: ", memory_usage, "MB")

    # Plot image and caption pairs with the cosine similarity between them
    plot_outputs(args.data_path, orig_imgs, cptns, similarity)
    # Plot cosine similarity between each pair of image and captions 
    plot_matrix(args.data_path, similarity)
   


 








    