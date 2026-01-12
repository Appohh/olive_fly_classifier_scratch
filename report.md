The report must be in imrad

# 1. Introduction

Olive flies are among one of the many species of flies that live around plants and stocks. They are considered as being a pest to the olives. That is why it can be benifitial to olive farmers to identify the flies on their stocks. There are however many more types of flies that exist on the olives that might not be harmfull to the plants. Therefore a method of just counting every fly would not work as that would result in false data and potentially incorrect and inadequate pest control. In order to be able to better control the flies and protect the olives, a farmer must know roughly how many olive flies there actually are on the olives.

This project aims to answer the question:

> How can a logistic regression model be constructed that is capable of identifying olive flies?

This kind of classification model is a perfect example of multiple logistic regression and a good first project for new learners. The project involves important fundemental steps like data pre-processing, model construction and training and model evaluation. This report describes and outlines the process of developing such a machine learning model. It aims to outline the methods used to pre-process the data and construct the model itself, the results of the model and closes with an analysis and discussion about pre-processing and training alternatives and approaches.

# 2. Methods

## 2.1. Data collection and pre-processing

The dataset consists of two image classes provided as part of the assignment: images containing olive flies and images containing other insects or objects found on olive plants. All images were resized to a fixed resolution of 170 × 170 pixels to ensure consistent input dimensions.
Pre-processing focused on isolating the insect from the background. Foreground extraction was performed using grayscale conversion followed by inverse Otsu thresholding, assuming the insect appears darker than the background. Morphological closing was applied to reduce noise, after which the largest connected component was selected as the foreground region. All non-foreground pixels were replaced with a white background.
To remove poorly extracted images, the ratio of foreground pixels to total pixels was computed for each image. Images exceeding a manually chosen threshold were considered faulty and removed from the dataset.
Data augmentation was applied to the olive fly class using horizontal flipping and small rotations of ±15 degrees to increase variation. After cleaning, the dataset was rebalanced by augmenting the olive fly class and randomly downsampling the non-olive fly class so that both classes contained an equal number of images.

## 2.2. Feature extraction

Histogram of Oriented Gradients features were extracted to capture shape and edge information of the insects. Images were converted to grayscale and normalized before computing HOG descriptors with fixed orientation bins, cell size, and block size.
Color information was captured using normalized three-dimensional RGB color histograms. Each image was converted to RGB space, and a fixed number of bins per channel was used to represent color distribution.
In addition to using HOG and color features separately, a combined feature representation was created by concatenating both feature vectors. This allowed the model to leverage both structural and color-based information during classification.

## 2.3. Model training

### 2.3.1. Sigmoid function

The nature of the problem is a binary classification problem. An object in an image must be identifed a "olive fly" or "not olive fly". Because of this very nature of the problem, a logistic regression model is chosen for the classification task. A logistic regression model is wel suited for these types of binary classification because it outputs a value between $0$ and $1$. Therefore we can apply the following logic:

$$
\hat{y} =
\begin{cases}
1, & \text{if } f(x) \ge 0.5,\\[4pt]
0, & \text{if } f(x) < 0.5.
\end{cases}
$$

A linear regression model works using the sigmoid function. This function essentially squeezed the output of the model between $0$ and $1$ by using the logarithm. It can be noted as

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Here, $z$ is essentially our multiple linear regression function $f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$

In the code, this sigmoid function is implemented using the numpy library.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### 2.3.2. Forward propogation

Of course, when the model is not yet trained, we do not know the $\beta_{1..n}$. However, to predict an outcome we must solve the multiple linear regression problem, to afterwards feed it to the sigmoid function in order for it to output the a prediction between $0$ and $1$. This value can then be interpreted as $true$ or $false$ and we have our prediction.

Mathmatically, this is represented as:

$$
\begin{aligned}
z &= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n \\
\sigma(z) &= \frac{1}{1+e^{-z}}
\end{aligned}
$$

And thats actually everything that is to the forward propagation from a mathematical perspective. There is however a big optimization technique that we can use to speed up the multiple linear regression calucation. Because all of the dependend variables $X = \set{x_1, x_2, \dots, x_n}$, independent variables $Y=\set{y_1, y_2, \dots, y_n}$ and the weights and bias can all be represented using matrices.

$$
X = \begin{bmatrix}
& x_{11} & x_{12} & \cdots & x_{1m} \\
& x_{21} & x_{22} & \cdots & x_{2m} \\
& x_{31} & x_{32} & \cdots & x_{3m} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
& x_{n1} & x_{n2} & \cdots & x_{nm}
\end{bmatrix}, \qquad
Y = \begin{bmatrix}
y_1 \\ y_2 \\ y_3 \\ \vdots \\ y_n
\end{bmatrix},
\qquad
\beta = \begin{bmatrix}
\beta_1 \\ \beta_2 \\ \beta_3 \\ \vdots \\ \beta_m
\end{bmatrix}
\qquad
\beta_0 = 0.0
$$

We can then use clever matrix operations to calculate the predictions in a batch, so we can calculate all the prediction at once using these matrix operations. We can calculate $\hat{y}$ by taking the dot product of $Y = X \times \beta + \beta_0$ And then we can directly feed in the result into the sigmoid function. In code this is implemented as follows

```python
Y_hat = sigmoid(np.dot(X, w) + b)
```

### 2.3.3. Gradient decent & backward propagation

With weights and biases of $0$ we cannot make accurate predictions. For that we need to train the model. Because at

  <Describe Describe gradient decent and backward propagation>
  <Describe Weights and Bias optimization through n itterations>

# Results

<Describe the model was trained on 80% of the data, and evaluated on 20%>
<Describe training params, and why they are good>
<Describe the different scores per model with different feature algorithms>
<Describe the training and test accuracy>
<Describe Precision, recall and F1 score>

# Analysis & Discussion

<Describe what the results mean>
<Describe the impact of the different feature algorithms>
<Describe other alternative feature generations>
<Describe impact of potential PCA (dimensionality reduction) usage>

# Conclusion

<Describe what this project did once more>
<Describe what the results mean for the reseach question>
<Answer the research question>

# Bibliography
