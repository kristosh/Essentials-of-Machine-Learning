---
layout: default
title: From linear models to Support vector machine and to kernels
description: A tutorial on kernels and Support vector machines
---

# The usual scenario in linear models classification

We have seen that usually in our classification problem we have datasets with the following form:

$$
\mathcal{D} = \{ \mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_n \} \in \mathbb{R}^{N \times D}
$$

where each instance $\mathbf{x}_n \in \mathbb{R}^D$ and since we are dealing with a supervised problem they are coupled together with annotation information which for this section we will assume that is binary annotation $(+1, -1)$ that corresponds in two classes. For instance, spam $+1$ or not spam $-1$. We can denote the full dataset with the the following tuples:

$$
 \mathcal{D} = \{ (\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \cdots, (\mathbf{x}_n, y_n) \} 
$$

The idea in linear models is to introduce parameters $\mathbf{w}$ that maps input data instances to the desired annotation. Thus, to create a program that learns these parameters in such a way that we have learned how the mapping between instances and labels is taking place. We saw, that the most simple way, the linear classification, we assume that this mapping can be revealed by performing linear operation only such as:

$$
\hat{y}_n = \mathbf{w}^T \mathbf{x}_n + b \in \mathbb{R}
$$

Note that if our input instances live in $\mathbb{R}^2$, thus, parameter $\mathbf{w} \in \mathbb{R}^2$ the above linear equation defines a plane. When we have $y_n = 0$ we can define a line which is the decision boundary that separates the two classes. If $y_n > 0$ then we can say that the instance belongs to the positive class while it will belong to the negative class otherwise.

For a specific instance $\mathbf{x}_n$, if we apply parameters $\mathbf{w}, b$ then we can compute $\hat{y}_n$ and our assumption is that we can learn such parameters that the prediction is as close as possible to the real label $y_n$. That is the idea behind linear model such as `Perceptron`. For this algorithm, there is a proof that if a dataset is linear separable then, `Perceptron` algorithm will always convergence and compute $\mathbf{w}, b$ that can separate the two classes. However, not all the datasets are linearly separable. For instance:

<p align="center">
  <img src="images/non_linear_1.png" alt="Sublime's custom image" style="width:50%"/>
</p>

In the above image, we can see a dataset of instances that live in the two dimensions (`feature A`, `feature B`). With red we represent the positive class while with blue the negative class. The purple instance is the item that we would like to classify in one of the two classes. If we try to learn parameters $\mathbf{w}, b$ using `Perceptron` or another linear strategy, we will figure out that is impossible to find a line that separates these two classes. 

<p align="center">
  <img src="images/non_linear_2.png" alt="Sublime's custom image" style="width:50%"/>
</p>

The same is happening in the above example.

However, while it is not feasible to find a linear decision boundary that separates perfectly this dataset, what can thing of 
a way to transform these data in such a way that then, it will be easy to find a linear decision boundary.

But how exactly can we transform the data from the last image. If you closely look the two classes, the `red-class` lives within the rectangular that is defined by points $(-0.5, -0.5)$ and $(.5, .5)$ while `the blue class` lives in the outer space from the rectangular that is defined by points $(-1, -1)$ and $(1, 1)$. 

Now instead of using the data directly as they are, we can firstly, apply a square function for each of the input features. Thus, each input instance $\mathbf{x} = [x_1, x_{2}]\in \mathbb{R}^2$ can be represented as $\mathbf{x}' = [x_1^{2}, x_{2}^{2}] \in \mathbb{R}^2$ with $(x_1, x_2)$ to be (`feature A`, `feature B`). This mapping now changes the data instances as follows:

<p align="center">
  <img src="images/non_linear_3.png" alt="Sublime's custom image" style="width:50%"/>
</p>

The reason that this is happening is really intuitive, every feature of the `red class` has absolute value that is less than one, while for `blue class` every instance has feature-values that are higher than one. Thus, when we square the features of these instances, `red class` is pushed closer to the origin, while `blue class` is pushed further. Note that since we squared all features, all feature-values for both classes are now positives. 

Now in this example, it is really easy to think of parameters $(\mathbf{w}, b)$ that define a decision boundary that perfectly separates the two classes. In this case, to learn this decision boundary, we first need to apply the transformation function or alternatively called the `basis function` and then, we can learn a decision boundary by simple using 
a linear model like `Perceptron`. Usually, we denote these basis functions as $\phi(\mathbf{x}) = \mathbf{x}'$. Now, the linear equation can be defined as:

$$
\hat{y}_n = \mathbf{w}^T  \phi(\mathbf{x}_n) + b \in \mathbb{R}
$$

while our dataset can be defined now as:  

$$
\mathcal{D} = \{ \phi(\mathbf{x}_1), \phi(\mathbf{x}_2), \cdots, \phi(\mathbf{x}_n ) \} \in \mathbb{R}^{N \times D} 
$$

Note that in the previous example $\mathbf{x}_n \in \mathbb{R}^2$ and $\phi(\mathbf{x}_n) \in \mathbb{R}^2$ so the mapping $\phi$ takes an input in two-dimensions and returns a transformed version of input into the same two-dimensional space, by just squaring each features. However, there is nothing that can constraints us to keep the same dimensionality. As a generalization we can consider that a `basis function` maps input space into another space $\phi: \mathbb{R}^{D} \to \mathbb{R}^{M}$. This can be seen in the following transformation:


<p align="center">
  <img src="images/non_linear_4.png" alt="Sublime's custom image" style="width:70%"/>
</p>

To sum up, we saw so far that when we do have a linear separable dataset, we can apply directly a linear model to learn a decision boundary the perfectly separates the two classes. When
our problem is not linear (meaning that we cannot compute a linear decision boundary to separate the two-classes) there is a solution to our problem. This is to apply a `basis function` that maps our initial dataset in such a way that we can simple then, apply a linear model to find a perfect decision boundary.

However, the example that we have presented, it was simple to find this basis function by just looking the image with all the instances of the dataset. Most of times, finding a good basis function is not an intuitive task and it requires extensive experimentation. In some other cases, the only way to achieve linearity is by applying `basis function` that lives in `a really high dimensional world`. Thus, it is not always easy or feasible to find a good `basis function`. Later in this tutorial we will introduce methods to mitigate this.

# Support Vector Machines

We have established already the fact that when our dataset is linear separable we can compute a decision boundary that is defined by parameters $(\mathbf{w}, b)$ via Perceptron algorithm. As you have seen during lectures, due to randomness in the algorithm, each time we run the algorithm we end-up having different decision boundary. That outcome can be seen in the following picture:

<p align="center">
  <img src="images/perceptron.png" alt="Sublime's custom image" style="width:50%"/>
</p>

There are multiple decision boundaries that are defined from different parameters $(\mathbf{w}, b)$ that perfectly separate a linear dataset or a non-linear datasets that we have applied some `basis function`. Which one of all these lines should we pick?

In perceptron, we can end up having any one these lines based on how we initialize our parameters. However, that is not a very good strategy. We can end-up having parameters that are very close to the `extreme` training samples of the`red` or `blue` class. We use the term `extreme` to denote those instances between the two classes that are close to the opposite class.

In this case, when we will deploy the model in the real-world application, there might be that we will need to classify some samples very close to the to `red` or `blue` class. It seems a bit unnatural to consider that the decision boundary should be really close to one of the two class. A more solid way to pick a decision boundary is to find the line that perfectly separates both classes and at the same time the distance between both classes (thus both extreme points from two classes) is equal. In this case, we can the distance of these
extreme points to the decision line `margin`.

## Linear models

The main idea behind many classification algorithms is to represent data in $\mathbb{R}^D$ and then partition this space, ideally in a way that examples
with the same label (and no other examples) are in the same partition. In the case of binary classification, the space would be divided into two
parts corresponding to the positive and negative classes, respectively. We consider a particularly convenient partition, which is to (linearly) split
the space into two halves using a hyperplane. Let example $\mathbf{x} \in \mathbb{R}^D$ be an element of the data space. Consider the function:

$$
f: \mathbb{R}^D \to  \mathbb{R}
$$

with function $f(\mathbf{x})$ to be:

$$
f(\mathbf{x}) = \mathbf{w}^T\mathbf{w} + b
$$

Therefore, we define the hyperplane that separates the two classes in our binary classification problem as:

$$
\{\mathbf{x} \in \mathbb{R}^D: f(\mathbf{x}) = 0 \}
$$

Since the input space is $\mathbb{R}^D$, parameters $(\mathbf{w}, b)$ define a hyperplane in $\mathbb{R}^{D-1}$ space. When the input is two-dimensional the decision boundary is a line, when the input three-dimensional that is a plane. An illustration of this function can be seen in the following figure:

<p align="center">
  <img src="images/hyperplane.png" alt="Sublime's custom image" style="width:50%"/>
</p>

From this example it is clear that $\mathbf{w}$ represents the slope of the hyperplane and the term bias $b$ the offset (the distance of the hyperplane from the origin). 

Vector $\mathbf{w}$ is always orthogonal to the hyperplane, and the hyperplane spans the space into two regions. In the first we have the first class while in the second we do have the second class.