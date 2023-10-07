# Decompositions

## Matrix Decomposition


  
Matrix decomposition, also known as matrix factorization, is a process of breaking down a matrix into simpler components that can be used to simplify calculations, solve systems of equations, and gain insight into the underlying structure of the matrix. 
  
Matrix decomposition plays an important role in machine learning, particularly in the areas of dimensionality reduction, data compression, and feature extraction. For example, Principal Component Analysis (PCA) is a popular method for dimensionality reduction, which involves decomposing a high-dimensional data matrix into a lower-dimensional representation while preserving the most important information. PCA achieves this by finding the eigenvectors and eigenvalues of the covariance matrix of the data and then selecting the top eigenvectors as the new basis for the data.

Singular Value Decomposition (SVD) is also commonly used in recommender systems to find latent features in user-item interaction data. SVD decomposes the user-item interaction matrix into three matrices: a left singular matrix, a diagonal matrix of singular values, and a right singular matrix. The left and right singular matrices represent user and item features, respectively, while the singular values represent the importance of those features.

Rank optimization is another method that finds a low-rank approximation of a matrix that best fits a set of observed data. In other words, it involves finding a lower-rank approximation of a given matrix that captures the most important features of the original matrix.  For example, SVD decomposes a matrix into a product of low-rank matrices, while PCA finds the principal components of a data matrix, which can be used to create a lower-dimensional representation of the data.   In machine learning, rank optimization is often used in applications such as collaborative filtering, image processing, and data compression. By finding a low-rank approximation of a matrix, it is possible to reduce the amount of memory needed to store the matrix and improve the efficiency of algorithms that work with the matrix.

We start with the eigenvalue decomposition (EVD), which is the foundation to many matrix decomposition methods

## Eigenvectors and eigenvalues  

Eigenvalues and eigenvectors have many important applications in linear algebra and beyond. For example, in machine learning, principal component analysis (PCA) involves computing the eigenvectors and eigenvalues of the covariance matrix of a data set, which can be used to reduce the dimensionality of the data while preserving its important features. 

TBA

## Singular Value Decomposition

Singular Value Decomposition (SVD) is another type of decomposition. Different than eigendecomposition, which requires a square matrix, SVD allows us to decompose a rectangular matrix. This is more useful because the rectangular matrix usually represents data in practice.

TBA
  
## Moore-Penrose inverse 

TBA
  
