
---
layout: default
title: Advanced Mathematics for Machine learning
Advanced Linear Algebra: Vector spaces, definition and examples
---

This page contains auxiliar material for this course. It contains definitions about concepts that are supplementary and optional for the EoML course. It only means for the curious reader and it does not need to be read for the course sake.

It contains advanced concepts for ML like vector spaces, linear independence, basis, basis transformation and matrix decomposition. 

## Vector spaces

In the beginning of this chapter, we informally characterized vectors as
objects that can be added together and multiplied by a scalar, and they
remain objects of the same type. Now, we are ready to formalize this,
and we will start by introducing the concept of a group, which is a set
of elements and an operation defined on these elements that keeps some
structure of the set intact.

A vector space is a really important definition in Linear Algebra but sometimes hard to understand and grasp. It is somehow simple but at the same time really abstract and vague. What a beauty! To grasp the reason that we need to introduce the vector space lets for a second focus or the real physical space of three-dimensions. Vector in physical space could represent quantities with both magnitude and direction. An example of that can be force, a velocity, gravity etcetera usually depicted with arrows. Length shows magnitude and the arrowhead shows direction, forming the basis of physics for describing motion and interactions. This physical space is equipped with two simple operators addition and scaling. If you add two vectors, or if you scale any vector you end-up having another vector within this physical space.

A vector space, it is an extension of the idea of physical space, but with objects (vectors) that they do not represent directly a physical meaning (like velocity, gravity etcetera) but could simple be a collection of number (data) as we have seen before and work similar with the physical spaces. They are equipped with the same properties (addition and scaling).

The reason why we study vector spaces is because they provide a useful framework for representing and solving many problems in mathematics, physics, engineering, and computer science. For example, they are used in linear algebra to study linear equations and matrices, in computer graphics to represent shapes and animations, and in machine learning to represent data.

As simple metaphors think about the LEGO blocks that could represent different vectors. The vector space is the entire collection of possible structures you can build using a set of initial LEGO blocks given to you in the box. Where where adding two structures gives another valid structure, and making a structure bigger or smaller also results in a valid structure. Another example to grasp what is going on here is with the computer graphics: A color in a game (like RGB values) can be represented by a vector, and the space of all possible colors is a vector space. 


 I will try to explain it in a very simple and intuitive way but skipping a lot of details. We will start with a simple example. Let's say that we got two vectors $\mathbf{v}_1 = [1, 1] \in \mathbbb{R}^{2}$ and $\mathbf{v}_2 = [1, 2]  \in \mathbbb{R}^{2}$. These vectors are a placeholder for two different `features` or we can call them `dimentions`, thus we say that these features belong to the $ \in \mathbbb{R}^{2}$ space. That means that these features lives in the space where all two-dimensional vectors that take real values live. That space can be easily represented by the cartesian space as such:

<p align="center">
  <img src="images/vectors.png" alt="Sublime's custom image"/>
</p>

### Vector sub-spaces

## Linear independence

## Basis and Rnak

## Linear mappings

## Basis change

## Affine spaces

## Affine mappings