---
layout: default
title: Introduction to Machine Learning .
description: Dictionary on Machine Learning
---

This page contains auxiliar material for this course. It contains definitions about concepts that are supplementary and optional for the EoML course. It only means for the curious reader and it does not need to be read for the course sake.

It contains advanced concepts for ML like vector spaces, linear independence, basis, basis transformation and matrix decomposition. 



# Vector redefined

As we have seen in the previous page a vector in computer science can be considered to be simply be an ordered list of numbers or things that can be eventually modelled and represented by numbers (text, audio, images etc). For example, when studying analytics on housing prices and it the only features that we care is the square footage and prices, so we can have as a vector:  $\mathbf{h}_1 = [55sq2, 500000] \in \mathbb{R}^{2}$, where house the first house is 55 square meters and costs 500.000€. In this case, the vectors happens to be in two-dimensional space.

However, there is nothing in the computer science that limits a vector to lie in just two dimensions. If we have a ordered list of n-elements (for instance features for housing) then we can talk about n-dimensional vectors. From the geometry perspective we usually talk about 2-dimensional or even 3-dimensional vectors that represents the coordinates that we physically live in, however, in algebra we can make the abstract leap can discuss about multiple dimensions.

## Vector spaces

A vector space is a really important definition in Linear Algebra and machine learning but sometimes hard to understand and grasp. I will try to explain it in a very simple and intuitive way but skipping a lot of details. We will start with a simple example. Let's say that we got two vectors $\mathbf{v}_1 = [1, 1] \in \mathbb{R}^{2}$ and $\mathbf{v}_2 = [1, 2]  \in \mathbb{R}^{2}$. These vectors are a placeholder for two different `features` or we can call them `dimentions`, thus we say that these features belong to the $ \in \mathbb{R}^{2}$ space. That means that these features lives in the space where all two-dimensional vectors that take real values live. That space can be easily represented by the cartesian space as such:

<p align="center">
  <img src="images/vectors.png" alt="Sublime's custom image"/>
</p>

Here we can see just two vectors in two dimesions, but in the vector space of two-dimensional real vectors you can imagine all the possible vectors in the cartesian space.

The same goes with the vectors belongs to three-dimensional space like $\mathbf{v}_1 = [1, 1, 0] \in \mathbb{R}^{2}$ and $\mathbf{v}_2 = [1, 2, 1]$. In this case, we will need to visualize the vectors in three-dimensions 

### example with three dimensional vectors

A vector space is a collection of things called vectors, where you can:

Add any two vectors and get another vector in the space.

Multiply any vector by a number (scalar) and still get another vector in the space.

### Vector sub-spaces

## Linear independence

Linear independence it is an important feature of vectors and a set of vectors which states the following: Having a set of vector $\{ \mathbf{x}_1, \mathbf{x}_2, \cdots \mathbf{x}_n  \}$ with $\mathbf{x}_i \in \mathbb{R}^{d}$, the vectors are linearly independent if and only if:

$a_1 \cdot \mathbf{x}_1 + a_2 \cdot \mathbf{x}_2 + \cdots + a_n \cdot \mathbf{x}_n = 0$ that is only the case when $a_1 = a_2 = \cdots = \a_n$

## Basis and Rnak

## Linear mappings

## Basis change

## Affine spaces

## Affine mappings