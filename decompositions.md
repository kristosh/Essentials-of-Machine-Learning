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

Let's assume a matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$, we can write the determinant of this matrix as follows:

$$
\det(A) = \begin{vmatrix}
a_{11} & a_{12} & \ldots & a_{1n} \\
a_{21} & a_{22} & \ldots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \ldots & a_{nn}
\end{vmatrix}
$$

Determinant is a function that maps matrix into a scalar value $\mathbb{R}$. The determinant is the entity that we use to check whether a matrix is invertible. It holds
that if a matrix $\boldsymbol{A}$ is invertilbe then $det(A) \neq 0$.

The notion of a determinant is natural when we consider it as a mapping from a set of $n$ vectors spanning an object in $\mathbb{R}^n$. 
It turns out that the determinant $\det(A)$ is the signed volume of an $n$-dimensional parallelepiped formed by columns of the matrix $A$. That can be seen with the following examples:

<p align="center">
  <img src="images/det1.png" alt="Sublime's custom image" style="width:20%"/>
</p>

<p align="center">
  <img src="images/det2.png" alt="Sublime's custom image" style="width:20%"/>
</p>

In the first image we do have two vectors $\mathbf{g} = [g, 0]^T$ and $\mathbf{b} = [0, b]^T$ and we can place them in the following matrix:

$$
\det(A) = \begin{vmatrix}
g & 0 \\
0 & b
\end{vmatrix}
$$

In this case, we define as determinant to be:

$$det(A) = g\cdot b + 0 =  g\cdot b $$

 which is the are of the parallelogram defined by the two vectors. The same happens in the second image, we can compute the area in the $\mathbb{R}^{3}$ space using the 
 determinant of the matrix that contains the vectors

 $$
\boldsymbol{A} = \begin{vmatrix}
\mathbf{g} & \mathbf{b} & \mathbf{r}
\end{vmatrix}
 $$

If for example we do have: 

$$r = \begin{bmatrix} 2 \\ 0 \\ -8 \end{bmatrix}, \quad g = \begin{bmatrix} 6 \\ 1 \\ 0 \end{bmatrix}, \quad b = \begin{bmatrix} 1 \\ 4 \\ -1 \end{bmatrix}$$

Then, the volume of these three vectors can be found in we compute $det(A)$:

$$
\det(A) = \begin{vmatrix}
2 & 6 &1\\
0 & 1 & 4\\
-8 & 0 -1&
\end{vmatrix} = 186
$$

One prerequisite is that the vectors are linearly independent otherwise we cannot compute the volume. Lets us image that the vectors $\mathbf{b}$ and $\mathbf{g}$ are dependant that means 
that they are parallel and thus, the area that they define is equal to zero.

<!-- The sign of the determinant indicates the orientation of the spanning vectors $\mathbf{b}, \mathbf{g}$ 
with respect to the standard basis $(\mathbf{e}_1, \mathbf{e}_2)$. In our figure, flipping the order to $\mathbf{g}, \mathbf{b}$ 
swaps the columns of $A$ and reverses the orientation of the shaded area. This becomes the familiar formula: area = height $\times$ length. 
This intuition extends to higher dimensions. In $\mathbb{R}^3$, we consider three vectors $\mathbf{r}, \mathbf{b}, \mathbf{g} \in \mathbb{R}^3$ 
spanning the edges of a parallelepiped, i.e., a solid with faces that are parallel parallelograms (see Figure 4.3). The absolute value of the 
determinant of the $3 \times 3$ matrix $[\mathbf{r}, \mathbf{b}, \mathbf{g}]$ is the volume of the solid. Thus, the determinant acts as a 
function that measures the signed volume formed by column vectors composed in a matrix. -->



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

In linear algebra, `eigensystems` denote a set of problems that include
finding eigenvectors and eigenvalues. The word eigen comes from
German and means `own`, which will make sense when we formulate the problem
more concretely. Informally, the main idea behind eigensystems is finding
vectors that transform in a special way when we apply a certain transformation
$\mathbf{A}$ on them. Specifically, we wish to find which vectors are affected
the least by the transformation $\mathbf{A}$, and by least we mean that they
are not rotated, but are only scaled by a factor $\lambda$. Formally, given a
vector $\mathbf{v}$ and a transformation $\mathbf{A}$, this requirement can be
written as:

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}

$$

Since on the right-hand side we multiply a vector by a scalar, we can
equivalently add the identity matrix as $\lambda \rightarrow \lambda \mathbf{I}$.
Rearranging terms gives us the following equation:

$$
(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = \mathbf{0}

$$

Assuming that $\mathbf{A} \in \mathbb{R}^{n \times n}$ and $\mathbf{v} \in \mathbb{R}^n$,
we can rewrite the former equation in an expanded form:

$$
\begin{pmatrix}
A_{11} - \lambda & A_{12} & \cdots & A_{1n} \\
A_{21} & A_{22} - \lambda & \cdots & A_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
A_{n1} & A_{n2} & \cdots & A_{nn} - \lambda
\end{pmatrix}
\begin{pmatrix}
v_1 \\
\vdots \\
v_n
\end{pmatrix}
=
\begin{pmatrix}
0 \\
\vdots \\
0
\end{pmatrix}
$$

The equation above represents a system of linear equations, and the goal is to
find vectors $\mathbf{v} = (v_1 \ \cdots \ v_n)^\top$ and $\lambda$ that satisfy
it. For example, the $m$-th equation is given by:

$$
A_{m1} v_1 + \cdots + (A_{mm} - \lambda) v_m + \cdots + A_{mn} v_n = 0.

$$

What we can see is that in every equation, we have all the unknowns (elements of
the vector). Therefore, if the equations are not linearly independent, the only
solution is the trivial one, i.e.\ $v_1 = v_2 = \cdots = v_n = 0$. This is a detrimental solution, and we are interested in the case that this vector is non-zero. 
In this case, it should hold that:

$$
|\boldsymbol{A} - \lambda \boldsymbol{I}| = 0
$$

### Geometrical interpretation and an example

So what exactly is an eigenvector from the geometry perspective. We saw that the eigenvalue portrays the variance of the initial data to the new coordinate axis that is 
represented by each eigenvector. However, what exactly the direction of this eigenvector could tell us?

The geometric interpretation for eigenvector is that once we compute the eigenvectors $\mathbf{b_1}, \mathbf{b_2}, \cdots, \mathbf{b_m}$ of matrix $\mathbf{A}$, then these vectors are not affected by the transformation with the 
matrix $\boldsymbol{A}$ except by a stretching factor $\lambda_{1}, \lambda_{2}, \cdots, \lambda_{m}$ in each case.

Why is this important?

## Matrix diagonalization

Suppose that we do have a matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ and it has $n$ linearly independent eigenvectors. Then, we can place these eigenvectors in matrix $\boldsymbol{S}$. Then, the product $\boldsymbol{S}^{-1}\boldsymbol{A}\boldsymbol{S} = \boldsymbol{\Lambda}$ is a diagonal matrix with the diagonal elements to be the eigenvalues of matrix $\boldsymbol{A}$.  

$$\boldsymbol{S}^{-1}\boldsymbol{A}\boldsymbol{S} = \boldsymbol{\Lambda} = \begin{bmatrix}
\lambda_1 & & & \\
& \lambda_2 & & \\
& & \ddots & \\
& & & \lambda_n
\end{bmatrix}$$

The proof is simple, if we stick to the product $\boldsymbol{A}\boldsymbol{S} $ we got:

$$
\boldsymbol{A}\boldsymbol{S} = A \begin{bmatrix}
| & | & & | \\
x_1 & x_2 & \cdots & x_n \\
| & | & & |
\end{bmatrix} = \begin{bmatrix}
| & | & & | \\
\lambda_1 x_1 & \lambda_2 x_2 & \cdots & \lambda_n x_n \\
| & | & & |
\end{bmatrix}
$$

Then the trick is to split this last matrix into a quite different product 

$$
\begin{bmatrix}
| & | & & | \\
\lambda_1 x_1 & \lambda_2 x_2 & \cdots & \lambda_n x_n \\
| & | & & |
\end{bmatrix} = \begin{bmatrix}
| & | & & | \\
x_1 & x_2 & \cdots & x_n \\
| & | & & |
\end{bmatrix} \begin{bmatrix}
\lambda_1 & & & \\
& \lambda_2 & & \\
& & \ddots & \\
& & & \lambda_n
\end{bmatrix}
$$

thus we can get:

$$
\boldsymbol{A}\boldsymbol{S} = \boldsymbol{S}\boldsymbol{\Lambda} 
$$

and finally its really easy to get:

$$

\boldsymbol{A} = \boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1}
$$

To grasp the importance of diagonalization we can have a view to the following example:

If we want to compute $\boldsymbol{A}^{n}$ we can simplify the computations as follows:

$$
\mathbf{A}^n = \underbrace{\mathbf{A} \cdots \mathbf{A}}_{n \text{ times}}
$$

and by replacing $\boldsymbol{A} $ with $\boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1}$ we do have:

$$
\mathbf{A}^n = \underbrace{(\boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1}) \cdots (\boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1})}_{n \text{ times}}
$$

$$
= \boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1}\boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1} \cdots \boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1}\boldsymbol{S}\boldsymbol{\Lambda}\boldsymbol{S}^{-1}
$$

$$
= \boldsymbol{S}\boldsymbol{\Lambda}^{n}\boldsymbol{S}^{-1}
$$

since all the intermediate steps $\boldsymbol{S}^{-1}\boldsymbol{S} = \boldsymbol{I}$. Now computing the $n$-th of the matrix can be computing by decompose the matrix into eigenvectors and eigenvalues
and computing the $n$-th power of the diagonal matrix which computational wise is an extreme simplification.


### Singular value decomposition

However, most of matrices are not square matrices and computing the diagonal is not as straight forward. One solution to this issue, is to perform Singular value decomposition (SVD). 

This technique is a generalization of matrix decomposition for non-square matrices and in ennesessence it tries to decompose the initial matrix $\boldsymbol{A} \in \mathbb{R}^{n \times m}$ In
both domain and subdomain. 

Then, the singular value decomposition is a factorization of the form:

$$
\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{T}
$$

The visualization of these matrices is shown below (adapted from Wikipedia). We can see that the matrix $\boldsymbol{\Sigma}$
 has a diagonal part (which can have zero and non-zero elements), whereas the rest of the matrix is equal to zero.

 <p align="center">
  <img src="images/svd_1.png" alt="Sublime's custom image" style="width:50%"/>
</p>

The diagonal elements $\sigma_i = \Sigma_{ii}$ of the matrix $\Sigma$ are called
`singular values` of $\mathbf{A}$. We can show that the number of nonzero
singular values is equal to the rank of $\mathbf{A}$. From decompositions
explained in the Rank of a Matrix theory page, we can see that the SVD
expression is equivalent to the following:

$$
\mathbf{A} = \mathbf{U} \Sigma \mathbf{V}^{\top}
= \sum_{i=1}^{\min(m,n)} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^{\top}
$$

From this, we see that the SVD of matrix $\mathbf{A}$ expresses it as a
(nonnegative) linear combination of rank-1 matrices, and we know that the
number of nonzero terms in such a linear combination is equal to the rank of
the matrix.

Geometrically, SVD actually performs very simple and intuitive operations.
Firstly, the matrix $\mathbf{V}$ performs a rotation in $\mathbb{R}^n$.
Next, the matrix $\Sigma$ simply rescales the rotated vectors by a singular
value and appends/deletes dimensions to match the dimension $n$ to $m$.
Finally, the matrix $\mathbf{U}$ performs a rotation in $\mathbb{R}^m$.

In the case of a real matrix, SVD can be visualized as shown below
(adapted from Wikipedia). On the top route, we can see the direct application
of a matrix $\mathbf{A}$ on two unit vectors. On the bottom route, we can see
the action of each matrix in the SVD. We have used a case of a square matrix,
as it is easier to visualize (in general, the matrix $\Sigma$ would add or
remove dimensions, depending on the form of the matrix $\mathbf{A}$).

 <p align="center">
  <img src="images/svd_2.png" alt="Sublime's custom image" style="width:50%"/>
</p>


### Dimensionality Reduction with Principal Component Analysis

Ok so far we saw multiple ways to describe a matrix with data $\boldsymbol{A}$. We learn about how to compute determinants and the trace of a matrix. We saw also 
how to perform an eigen-analysis of the matrix and what is the geometric interpretation of it. We saw how to diagonalize a matrix and how to do it using Singular value 
decomposition. In this part of the tutorial, we will see its real merit and the reasons why we would like to perform matrix decomposition in Machine Learning. Plot-twist
the main reason is because matrix decomposition paves the way for dimensionality reduction and the discovery of embeddings that can be meaningfully characterize the 
initial feature space of our data in hand.

In ML the most interesting and challenging problems are coupled with data that live in high-dimensionalities such as images. This high-dimensionality comes 
with multiple problems such as it makes the ML algorithm hard to parse data to interpret them while it is merely impossible to visualize them and really expensive to 
store the data in servers. At the same time, there are properties of these high-dimensional data that we can take advantage of. For instance, many dimensions are redundant and they 
could simply represented a linear combination of other dimensions. Dimensionality reduction exploits structure and correlation and allows us to work with a more compact representation
of the data, ideally without losing information. We can think of dimensionality reduction as a compression technique, similar to jpeg or mp3, which are compression algorithms 
for images and music.


### Principle component analysis

We will start with a simple intuition that I like to use when explaining PCA. Conceptually the way that PCA works reminiscence the way that the photography works in the physical space of three
dimensions. Imagine the following setup, we do have living persons in the physical world of three dimensions $(x, y, z)$, width, length and height. We would like to collect images in such as way
that we will represent perfectly all the living persons in the scene. One option since our problem lives in 3d would be to collect data from three different axis. However, in practice
that is not necessary and a photographer does is to find the perfect angle in each he can perfectly capture the ideal information for all the subjects in the scene. In the same spirit,
we can perceive PCA as finding a new angle to capture our data which better characterize the initial information from our data. 

<p align="center">
  <img src="images/pic.png" alt="Sublime's custom image" style="width:50%"/>
</p>


<p align="center">
  <img src="images/pic1.png" alt="Sublime's custom image" style="width:50%"/>
</p>


<p align="center">
  <img src="images/pic2.png" alt="Sublime's custom image" style="width:50%"/>
</p>


But the intuition sounds sweet, however, how this is done in practice? In Principle component analysis (PCA) we are interested in finding projections of the initial data $\mathbf{x}_n \in \mathbb{R}^{D}$ denoted as $\mathbf{\tilde{x}}_n \in \mathbb{R}^{D'}$ which are as close as
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