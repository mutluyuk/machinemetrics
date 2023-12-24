
# Graphical Network Analysis

  
A network represents a structure of relationships between objects. Graphical modeling presents a network structure in a graph, which consists of nodes and edges, by expressing conditional (in)dependence between the nodes. If we think of these nodes (objects) as variables and their relationship with each other as edges, a graphical model represents the probabilistic relationships among a set of variables. For example, the absence of edges (partial correlations) corresponds to conditional independence. Graphical models are becoming more popular in statistics because it helps us understand a very complex structure of relationships in networks, such as the dynamic structure of biological systems or social events.  

The central idea is that, since any pair of nodes may be joined by an edge, a missing edge represents some form of independency between the pair of variables. The complexity in network analysis comes from the fact that the independency may be either marginal or conditional on some or all of the other variables.  Therefore, defining a graphical model requires identification of a type of graph needed for each particular case.

In general, a graphical model could be designed with directed and undirected edges. In a directed graph, an arrow indicates the direction of dependency between nodes. In undirected graphs, however, the edges do not have directions.  The field of graphical modeling is vast, hence it is beyond the scope of this book. 

Yet, we will look at the precision matrix, which has been shown that its regularization captures the network connections.  Hence, the central theme of this section is the estimation of sparse standardized precision matrices, whose results can be illustrated by undirected graphs.

## Fundementals

In this chapter, we will cover several concepts related to statistical (in)dependence measured by correlations.

## Covariance

TBA

## Correlation

TBA

## Semi-partial Correlation

With partial correlation, we find the correlation between $X$ and $Y$ after controlling for the effect of $Z$ on both $X$ and $Y$. If we want to hold $Z$ constant for just $X$ or just $Y$, we use a semipartial correlation.
  
While a partial correlation is computed between two residuals, a semipartial is computed between one residual and another variable. One interpretation of the semipartial is that the influence of a third variable is removed from one of two variables (hence, semipartial). T

TBA
  

## Regularized Covariance Matrix

Due an increasing availability of high-dimensional data sets, graphical models have become powerful tools to discover conditional dependencies over a graph structure.


 TBA

