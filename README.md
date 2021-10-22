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

The number `_,IVFxxxxxx_,_` defines how large your training set needs to be. 
The number of descriptors used in the training set lie between `30*xxxxxx` and `256*xxxxxx`.
In this project you just set the parameter `FRACTION_DES_USED_FOR_TRAINING` equal to `#training_des/#total_des`.

Make sure you have enough memory available for the program. 
Provide at least the size of `FRACTION_DES_USED_FOR_TRAINING * size_of_all_des * 2`.
In a Windows machine, you can increase the memory size by following the steps on:
https://www.windowscentral.com/how-change-virtual-memory-size-windows-10.



### Offline

Run `offline.py`. 
This wil generate an index and a list of keypoints of every descriptor of every image in the `data/` folder.
Specify the filename of the keypoint list and the index in `parameters.py`.

### Online

Run `online_debug.py` to get accuracy and performance results on all your images in the `testset/` folder or subfolders.

Run `online.py` to test single images using the command line.

## How it works

We're trying to find the closest match to a descriptor as fast as possible with as little storage as possible.
Without a clever data structure this would require calculating the distance between every vector and storing every vector on disk (=linear storage).

The faiss library developed by Facebook AI Research uses state-of-the-art data structures for nearest neighbour search.
Here we'll briefly (try to) explain the concepts behind LOPQ, one of the best performing datastructures at this time.
To better understand it, I highly recommend the following video: https://www.youtube.com/watch?v=RgxCaiQ-kig.
### Q

Quantisation is essentially just running k-means. 
Please read [_Scalable Recognition > How It Works > Scaling Up > K-means_](https://github.com/DaanS8/ScalableRecognition#k-means) to understand the algorithm.
K-means is a clustering algorithm, it assigns every vector to one of the k clusters. 
To search the nearest neighbour only the cluster centers need to be checked.

![LOPQ](img_rm/lopq_a.png)


- Asymptotic runtime of k-means: `O(N.k.d.i)`.
- Storage requirements: `N.log₂(k) + k.d.32b`. 
Which cluster center is closest + cluster centers, storing a vector requires a 32-bit float per dimension.

Here N is the total amount of vectors, k is the amount of cluster centers, d is the dimensionality of the vectors and i is the amount of iterations.
We have a large N, k and d. Running k-means is computationally expensive.
To keep track of which cluster center is closest to a descriptor, every descriptor only needs to keep track of `log₂(k)` bits.
This is a high compression rate: `N.log₂(k) + k.d.32b << N.d.32b`. 

Looking up which vectors are closest to which cluster center is slow if every vector needs to be checked.
Often an inverted file index is used, the image below gives an example of an inverted file index.
For every cluster center `q(x)`, a list of all descriptor ids that are closest to it is stored.
This drastically reduces lookup time.

![Inverted file list](img_rm/inverted_list.png)

### PQ

Product Quantisation improves k-means by reducing the amount of runtime needed for the same accuracy.
It works by running k-means on sub-vectors of the original vectors.
An example makes this way more clear, checkout the image below.
In the example the dimensionality is two, and we split the vectors up in two sub-vectors (`m=2`).
So we first run k-means with `k'=8` only on the x-values of the vectors, and afterwards on the y-values.
Running k-means twice with a k of 8 results in 8² possible combinations of where a vector can be closest to.

- Asymptotic runtime of PQ: `O(m.N.k'.d.i)`. 
- Storage requirements: `m.N.log₂(k') + k'.d.32b`. Storing cluster centers requires `m.(d/m).32b` per `k'`.

Why is this an improvement?
Take our case: `N=100_000_000, d=128, k=1_000_000, i=1` -> `N.d.k.i = 1.28 * 10¹⁶`.
With `m=4` to reach the same amount of cluster centers we only need `k'=32` because `32⁴ > 1_000_000`.
`m.N.k'.d.i = 1.28 * 10⁸`, 8 orders smaller than running k-means.
The cluster centers of k-means are more precise, but increasing k' for more cluster centers still takes way less time than normal quantisation.


![LOPQ](img_rm/lopq_b.png)

### OPQ

Optimised Product Quantisation reduces the distortion of PQ. 
The general approach of the algorithm is as followed:

- mean center the data 
- Perform Eigen decomposition 
- Use eigen allocation to compute permutation matrix P 
- Compute rotation matrix R= QPᵀ 
- Rotate all data by R⁻¹ 
- Perform Product Quantisation in rotated space

So we use Principal Component Analysis to rotate the data so that the distortion of PQ is minimised. 
This approach is directly taken form the recommended [video](https://www.youtube.com/watch?v=RgxCaiQ-kig&t=3431s), please watch it for more implementation details.

Note that calculating the rotation matrix for large N is computationally expensive, but it does increase performance.
![LOPQ](img_rm/lopq_c.png)

### LOPQ

Locally Optimised Product Quantisation further improves OPQ.
OPQ uses the (wrong) assumption that the data is normally distributed.
In the real world, f.e. SIFT vectors, are multimodal.

LOPQ uses a coarse quantizer, it runs k-means with a low K.
In every found cluster group, run OPQ using only vectors of that group.

If the initial k is sufficiently low, this is computationally doable.
Because of the lower amount of vectors in every group, calculating the rotation matrix is computationally doable as well.
So it uses k-means to group multimodal data and uses the performance benefits of OPQ to achieve great accuracy and performance.


![LOPQ](img_rm/lopq_d.png)

Here a final overview of the evolution of quantisation. 

![LOPQ](img_rm/lopq.png)

