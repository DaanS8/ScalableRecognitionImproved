# Scalable Recognition Improved
A python implementation for a scalable object detection framework. 
This is an improved version of the [Scalable Recognition repository](https://www.github.com/DaanS8/ScalableRecognition) that was based on the paper [Scalable Recognition with a Vocabulary Tree](https://ieeexplore.ieee.org/document/1641018). 
Due to the use of state-of-the-art datastructures for nearest neighbor search the efficiency and accuracy was improved dramatically.
F.e. on my dataset of 100,000 images the original repository achieved an accuracy of 85,5% with a process time of 1,33s/image.
This repository achieved an accuracy of 96.5% with a process time of 1.16s/image.

## Instalation

Conda Installation:

    conda install -c conda-forge opencv numpy-indexed faiss-gpu cudatoolkit=xx.x

How you can check your cuda version is explained [here](https://stackoverflow.com/questions/9727688/how-to-get-the-cuda-version).

## How to use

### Setup

Put all your database images in the `data/` folder. 
Every image should have the format id.jpg where id is an integer. 
To be complete, the ids aren't required to be sequential. 
The images in the data/ folder are often referred to as the db (database) images.

Put all your query images in the `testset/` folder. 
Using subfolders like `testset/Easy/` etc. is possible.
Every image must have a .jpg extension.
Optional: if you'd like to test the accuracy, then name every query image id.jpg where id is the id of the correct db image match.


Checkout which database structure you need. 
In the [wiki](https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors) of the faiss library benchmarks of different database structures using different vectors are given, under the menu item _Typical use cases and benchmarks_.
My database contained 200M SIFT vectors, so I've used `OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr` as my data structure because it's (one of) the best index structures for 100M SIFT vectors.




### Offline


### Online