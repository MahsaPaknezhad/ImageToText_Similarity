# Solution for the AI Team Challenge

For this challenge, we deployed [CLIP](https://openai.com/blog/clip/). CLIP (Contrastive Language-Image Pre-Training) is a deep learning model that is trained on a variety of (image, text) pairs. CLIP has both an image encoder and a language encoder. As a result of its novel training process, CLIP is able to map texts and images to the same latent space.

## Approach

We used CLIP to map the images and their corresponding captions to the same latent space and utilized cosine similarity to measure the similarity between the image and caption embeddings in the latent space. Figure shows our pipeline more clearly. 

![Pipeline](assets/image_text_similarity.png)


## Usage

We installed [PyTorch 1.8.1](https://pytorch.org/get-started/locally/), torchvision, as well as other additional dependencies as shown below. Then CLIP was installed as a Python package. The commands for installing the above packages on a CUDA GPU machine are as follows:

```bash
$ conda create -n clip_env  -c conda-forge -c nvidia python=3.10 cuda-version=11.2 pytorch torchvision
$ pip install ftfy regex tqdm pandas
$ pip install git+https://github.com/openai/CLIP.git
```

Replace `cudatoolkit=11.2` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

To run the code, activate the created ```clip_env``` venv and run the ```main.py``` python file. At the conclusion of the file's execution, the following outputs are generated:


