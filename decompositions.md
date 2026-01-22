---
layout: default
title: Matrix decompositions and eigen-analysis
description: Matrix decompositions and from eigen-analysis to Principal component analysis
---

# Matrix decompositions

What is it? Why we need it?

In general we have seen in previous tutorials how mappings and transformations of vectors can be conveniently described as operations performed by
matrices. We saw how data can be represented by matrices where the rows of the matrix represent different people and the columns
describe different features of the people, such as weight, height, and socioeconomic status. In this chapter, we present three aspects of matrices: how
to summarize matrices, how matrices can be decomposed, and how these decompositions can be used for matrix approximations.

We first consider methods that allow us to describe matrices with just a few numbers that characterize the overall properties of matrices. We
will do this in the sections on determinants (Section 4.1) and eigenvalues (Section 4.2) for the important special case of square matrices.

Matrix decompositions are important decompositions in linear algebra as they allow for simpler computations with matrices, 
as we shall see in the following subsections. On the other hand, they are also widely used in machine learning, 
for the purposes of data compression, denoising, and latent semantic analysis to name a few. Matrix decompositions usually decompose an original matrix 
into a product of simpler matrices, which have some specific features. In this theory page, we cover two important decompositions: 
Eigenvalue decomposition (Diagonalization) and Singular value Decomposition (SVD).

Firstly, we will start with trace of a matrix which is one aspect of describing a matrix.

## Determinant of a matrix

Determinants as measures of volumes.

## Trace of a matrix 

The trace of a matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ is defined as:

$$
tr(\boldsymbol{A}) = \sum_{i=1}^n a_{ii}
$$

where $a_{ii}$ are the diagonal elements of the squared matrix.

The trace satisfies the following properties:

- $tr(\boldsymbol{A} + \boldsymbol{B}) = tr(\boldsymbol{A}) + tr(\boldsymbol{B})$ for $\boldsymbol{A}, \boldsymbol{B} \in \mathbb{R}^{n \times n}$
-$tr(\alpha \boldsymbol{A}) = \alpha tr(\boldsymbol{A})$, $\alpha \in \mathbb{R}$ for $\boldsymbol{A} \in \mathbb{R}^{n \times n}$
-$tr(\boldsymbol{I}_n) = n$
- $tr(\boldsymbol{AB}) = tr(\boldsymbol{BA})$ for $\boldsymbol{A} \in \mathbb{R}^{n \times k}$, $\boldsymbol{B} \in \mathbb{R}^{k \times n}$


It can be shown that only one function satisfies these four properties together -- the trace (Gohberg et al., 2012).

The properties of the trace of matrix products are more general. Specifically, the trace is invariant under cyclic permutations, i.e.,

$$
tr(\boldsymbol{AKL}) = tr(\boldsymbol{KLA})
$$

for matrices $\boldsymbol{A} \in \mathbb{R}^{a \times k}$, $\boldsymbol{K} \in \mathbb{R}^{k \times l}$, $\boldsymbol{L} \in \mathbb{R}^{l \times a}$. This property generalizes to products of an arbitrary number of matrices. As a special case of (4.19), it follows that for two vectors $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n$

$$
tr(\boldsymbol{xy}^\top) = tr(\boldsymbol{y}^\top \boldsymbol{x}) = \boldsymbol{y}^\top \boldsymbol{x} = \boldsymbol{x}^\top \boldsymbol{y} \in \mathbb{R}
$$


## Eigenvalues and Eigenvectors

Let $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ be a square matrix. Then, the entity $\lambda \in \mathbb{R}$ is called an `eigenvalue` and the entity $\mathbf{x} in \mathbb{R}^n$ eigenvalue of matrix $\boldsymbol{A}$ if the following is true:

$$
\boldsymbol{A} \mathbf{x} = \lambda \mathbf{x}
$$

We can call this an `eigenvalue equation`.


### Singular value decomposition

## Matrix decomposition for Machine Learning

### Dimensionality Reduction with Principal Component Analysis

Working directly with high-dimensional data, such as images, comes with some difficulties: It is hard to analyze, interpretation is difficult, visualization
is nearly impossible, and (from a practical point of view) storage of
the data vectors can be expensive. However, high-dimensional data often
has properties that we can exploit. For example, high-dimensional data is
often overcomplete, i.e., many dimensions are redundant and can be explained
by a combination of other dimensions. Furthermore, dimensions
in high-dimensional data are often correlated so that the data possesses an
intrinsic lower-dimensional structure. Dimensionality reduction exploits
structure and correlation and allows us to work with a more compact representation
of the data, ideally without losing information. We can think
of dimensionality reduction as a compression technique, similar to jpeg or
mp3, which are compression algorithms for images and music.


### Principle component analysis

In Principle component analysis (PCA) we are interested in finding projections of the initial data $\mathbf{x}_n \in \mathbb{R}^{D}$ denoted as $\mathbf{\tilde{x}}_n \in \mathbb{R}^{D'}$ which are as close as
possible to the original point but at the same time lives in a dimensionality and is lower than the initial one $D' << D$.

Usually, the setup is the following: we do have access to a dataset $\mathcal{D} = \\{ \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n \\} \in \mathbb{R}^{N \times D}$ with $N$ to be the number of 
instances and $D$ the dimensionality of each feature with matrix $\boldsymbol{S}$ to be the data covariance matrix:

$$
\boldsymbol{S} = \frac{1}{N}\sum_{i=1} ^N \mathbf{x}_n \mathbf{x}_n^T
$$

Then in PCA we assume that there is a low dimension representation that is characterized by projection matrix $\boldsymbol{\mathcal{B}} = \[ \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_M \] \in \mathbb{R}^{D \times M}$ where you 
can project each instance $\mathbf{x}_n$ as:


$$
\mathbf{z}_n = \boldsymbol{B}^{T} \mathbf{x}_n
$$

Our target is to find the projection matrix $\boldsymbol{\mathcal{B}}$ that maps the original data with minimal compression loss while keeps the data information intact. In the below sections we will 
analyze two approaches to find this projection matrix. The first approach, aims at finding the direction that maximizes the variance, while in the second perspective, we are trying to minimize the reconstruction
loss.


#### First perspective maximizing the variance

Each vector $\[ \mathbf{b}_1, \mathbf{b}_2, \cdots, \mathbf{b}_M \]$ projects the initial data into a new coordinate space. The total amount of dimensions is $M$ and this is a hyperparameter that we
usually need to decide or to figure out which dimensionality is more suitable for the problem in hand. The ideas is also to sort these projections in such a way that that the first projection leads to the 
direction with the maximum variance and so on for the rest of the projections. If the dimensionality of the initial data is $D$ then we can find initially $D$ new projections that the variance is 
maximized and sorted in descending order. Thus, we will need to find these projection vectors that lead to directions where the variance of the data is sorted in this way. 


Initially, we start by trying to find the projection vector $\mathbf{b}_1 \in \mathbb{R}^M$ that maximizes the variance of the projected data. Usually, in PCA, we make the assumption that the data are centered around mean since the variance is not 
affected by this mean. That means that even if the data are not centered around the mean, if we subtract the mean and center the data, the variance is the same. Thus, we can compute the variance of the first coordinate as:

$$
V_1 = \mathbb{V}[z_1] =  \frac{1}{N}\sum_{i=1}^{N}z_{1n}^{2}
$$

where $z_{1n} = \mathbf{b}_{1}^{T} \mathbf{x}_n$ is the first coordinate that corresponds to the projection of $\mathbf{x}_n$ onto the one-dimensional space spanned by vector $\mathbf{b}_1$. Thus, we can compute
the total variance for this dimension based on the input data $\mathcal{D} = \\{ \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n \\} \in \mathbb{R}^{N \times D}$:

$$
V_1
= \frac{1}{N} \sum_{n=1}^{N} (b_1^\top x_n)^2
= \frac{1}{N} \sum_{n=1}^{N} b_1^\top x_n x_n^\top b_1 $$

$$
= \boldsymbol{b}_1^\top \left( \frac{1}{N} \sum_{n=1}^{N} x_n x_n^\top \right) \boldsymbol{b}_1
= \boldsymbol{b}_1^\top S \boldsymbol{b}_1 .
$$

We have expanded the variance and reach to the final expression by using the property that dot product is symmetric. To make things easy, in our search for the direction that maximizes the variance, we can assume without
that the magnitude of this vector is normalized to be equal to one, thus: $||\mathbf{b}_1|| = 1$ since what we care is the direction and not the length of the vector. 

This opens the gate for utilized and using constraint optimization techniques such as Lagrangian multipliers:

$$
\max_{\boldsymbol{b}_1} \; \boldsymbol{b}_1^{\top} S \boldsymbol{b}_1
$$

$$
\text{subject to } \|\boldsymbol{b}_1\|^2 = 1.

$$

This is an equality constraint optimization problem, so we can introduce the following Lagrangian:

$$
\mathcal{L} = \boldsymbol{b}_1^\top S \boldsymbol{b}_1 + \lambda_1 (1 - b_1^\top \boldsymbol{b}_1)
$$

The recipe for Lagrangian constraint optimization is to find the partial derivatives with respect to $\mathbf{b}_1$ and $\lambda_{1}$ which leads to the following outcome:

$$
\boldsymbol{S}\boldsymbol{b}_1 = \lambda_{1} \boldsymbol{b}_1
$$

It is obvious that by computing the partial derivatives we ended up in an equation system that is equal to eigen-decomposition and vector $\mathbf{b}_1$ is an eigenvector while 
parameter $\lambda_{1}$ is an eigenvalue. Now, if we dig further we do have the following expression:

$$
V_1 = \boldsymbol{b}_1^\top S \boldsymbol{b}_1  = \boldsymbol{b}_1^\top \lambda_{1} \boldsymbol{b}_1 =  \lambda_{1} \boldsymbol{b}_1^\top  \boldsymbol{b}_1  = \lambda_{1}
$$

What this equation express is that the to maximize the variance of this projection, we need to find the eigenvector with the largest eigenvalue of the data covariance matrix. Then,
this vector $ \boldsymbol{b}_1$ is called `first principal component`.

Once we found the first PC we can subtract this from the data matrix and perform the same process again until we find $M$ principal component vectors. While we will not present here all the details of how this
is done, in principle we can have:

$$
\boldsymbol{\hat{X}} = \boldsymbol{X}  - \sum_{i=1}^{m-1} b_i b_i^{T} \boldsymbol{X}  = \boldsymbol{X}  - \boldsymbol{B}_{m-1} \boldsymbol{X} 
$$

The above shows a generalization of how we can subtract the first $m-1$ principal components from our matrix $\boldsymbol{X}$.

We can proceed that way that we can compute $M$ principal components. We can even compute $M = D$ components. Once we compute all of them, we can sort the PC based on their eigenvalue so the
variance in the projected coordinate. Finally, we can keep all the coordinates that lead to minimum reduce of the variance in data with the respect to the variance of the initial data.

#### Projection perspective

In the following, we will derive PCA as an algorithm that directly minimizes the average reconstruction error which is a second perspective in PCA.


#### Example MNIST embeddings


#### How to compute eigenvectors and eigenvalues using Singular-value decomposition (SVD)

[back](./)