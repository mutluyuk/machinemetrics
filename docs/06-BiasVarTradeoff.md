
# Bias-Variance Trade-off 

In this chapter, we explore the bias-variance tradeoff and its critical role in model performance. In this chapter, we still assume prediction functions (i.e. we assume we know the prediction functions in simulations) to find MSPE using the given sample dataset. We use single data set to calculate MSPE for that given data set (in-sample MSPE for a test data). In this chapter, we show calculating bias-variance trade-off for each assumed function using the sample data. Bias-Variance Trade-off, while complex, is reflected in everyday life situations, offering insights into its practical significance. We begin with the story of Baran, whose experiences in planning his biweekly work trips vividly illustrate this tradeoff.

Baran aims to catch a flight for his biweekly work trips, with the gate closing at 8 AM. During his first year, he developed a routine to leave his house around 7:00 AM, planning to depart either 30 minutes before or after this time. However, he discovered that this approach sometimes led to him missing his flight. Even though his average or expected departure time from home was 7:00 AM, this process occasionally resulted in arrival times after the flight's gate closure, making his goal of catching the flight more error-prone. This approach represented a scenario with higher variance and lower bias, leading to costly errors concerning his airport arrival time and missed flights. In contrast, with his revised preparation routine, Baran aimed to leave around 7:00 AM but allowed a margin of 10 minutes either way. Consequently, his average departure time shifted to 7:05 AM, introducing a 5-minute bias. However, this new strategy led to more consistent success in catching his flight, reflecting higher bias but lower variance.  This approach nearly eliminated all instances of missing the flight. This whole experience demonstrates the bias-variance tradeoff. In the first case, a preparation process with higher variance and lower bias resulted in occasional flight misses. In the second case, adopting a process with reduced variance, even though it introduces a degree of bias, ultimately enhances the overall result. Baran's experience illustrates that consistently opting for lower bias (even unbiased), is not always the best strategy to predict his arrival time, as a balance between bias and variance can lead to more effective real-world results.

As seen from Baran's narrative, the bias-variance tradeoff is not an abstract statistical concept but a tangible factor in our everyday decision-making processes. Grasping this balance is key to understanding how to craft models that are not only accurate but also reliable across various situations. In the next chapter, we'll explore another critical concept in machine learning: overfitting. This concept is also very crucial for crafting precise and dependable models. We will explore how overfitting relates to and differs from the bias-variance tradeoff. This exploration will further enhance our understanding of model development, providing deeper insights into creating effective and reliable machine learning algorithms.

We already discuss bias and variance of estimator and predictor, and decomposition of MSE for estimation and MSPE for prediction in the previous chapter. Now, Lets discuss what is bias and variance and trade-off between them in predictive models. The bias-variance tradeoff is a fundamental concept in statistical learning and machine learning, which deals with the problem of minimizing and balancing two sources of error that can affect the performance of a model. Bias occurs from oversimplifying the model, leading to underfitting and missing relevant relations in the data. Variance occurs from overcomplicating the model, leading to overfitting and capturing random noise instead of the intended outputs. High bias results in poor performance on training data, while high variance causes poor generalization to new data. The goal is to find a balance, creating a model that is complex enough to capture true patterns but simple enough to generalize well. This balance is achieved through choosing appropriate model complexity, using techniques like cross-validation, applying regularization, and refining data or features.

Following is a formal discussion of bias-variance tradeoff. 

## Formal Definition

To remind,our task is prediction of an outcome, Y using the data (more accurately test/train data) .

To "predict" Y using features X, means to find some $f$ which is close to $Y$. We assume that $Y$ is some function of $X$ plus some random noise.

$$Y=f(X)+ \epsilon$$
However, we can never know real $f(X)$. Thus our goal becomes to finding some \(\hat{f(X)}\) that is a good estimate of the regression function $f(X)$. There will be always difference between real $f(X)$ and $\hat{f(X)}$. That difference is called reducible error. We find a good estimate of $f(X)$ by reducing the expected mean square of error of test data as much as possible. Thus, we can write 

**MSE for prediction function using training data** (for test data, validation sample):

$$
\operatorname{MSE}(f(x), \hat{f}(x))= \underbrace{(\mathbb{E}[f(x)-\hat{f}(x)])^2}_{\operatorname{bias}^2(\hat{f}(x))}+\underbrace{\mathbb{E}\left[(\hat{f}(x)-\mathbb{E}[\hat{f}(x)])^2\right]}_{\operatorname{var}(\hat{f}(x))}
$$

Then,  

**Mean Square of Prediction Error** can be written as:

$$
\text{MSPE}=\mathbb{E}\left[(Y-\hat{f(X)})^{2}\right]=(\text{Bias}[\hat{f(X)}])^{2}+\text{Var}[\hat{f(X)}]+\sigma^{2}
$$
$\sigma^{2}=E[\varepsilon^{2}]$

The expected mean-squared prediction error (MSPE) on the validation/training sample consists of the sum of the squared bias of the fit and the variance of both the fit and the error/noise term. The error term $\sigma^2$, also referred to as irreducible error or uncertainty, represents the variance of the target variable $Y$ around its true mean $f(x)$. It is inherent in the problem and does not depend on the model or training data. When the data generation process is known, $\sigma^2$ can be determined. Alternatively, estimation of $\sigma^2$ is possible using the sample variance of $y$ at duplicated (or nearby) inputs $x$.

**Variance** is the amount by which $\hat{f(X)}$ could change if we estimated it using different test/training data set. $\hat{f(X)}$ depends on the training data. (More complete notation would be $\hat{f}(X; Y_{train},X_{train})$ )If the $\hat{f(X)}$ is less complex/less flexible, then it is more likely to change if we use different samples to estimate it. However, if $\hat{f(X)}$ is more complex/more flexible function, then it is more likely to change between different test/training samples. 

For instance, Lets assume we have a data set (test or training data) and we want to "predict" Y using features X, thus estimate function $f(X)$. and lets assume we have 1-degree and 10-degree polynomial functions as a potential prediction functions. We say $\hat{f(X)}=\hat{\beta_{0}} + \hat{\beta_{1}} X$ is less complex than 10-degree polynomial $\hat{f(X)}=\hat{\beta_{0}} + \hat{\beta_{1}} X...+ \hat{\beta_{10}} X^{10}$ function. As, 10-degree polynomial function has more parameters $\beta_{0},..\beta_{10}$, it is more flexible. That also means it has high variance (As it has more parameters, all these parameters are more inclined to have different values in different training data sets). Thus, a prediction function has high variance if it can change substantially when we use different training samples to estimate $f(X)$. We can also say less flexible functions (functions with less parameters) have low variance. Low variance functions are less likely to change when we use different training sample or adding new data to the test sample. We will show all these with simulation in overfitting chapter as well.     

**Bias** is the difference between the real! prediction function and expected estimated function. If the $\hat{f(X)}$ is less flexible, then it is more likely to have higher bias. We can think this as real function(reality) is always more complex than the function approximates the reality. So, it is more prone to have higher error, i.e. more bias.  

In the context of regression, Parametric models are biased when the form of the model does not incorporate all the necessary variables, or the form of the relationship is too simple. For example, a parametric model assumes a linear relationship, but the true relationship is quadratic. In non-parametric models when the model provides too much smoothing.

There is a bias-variance tradeoff. That is, often, the more bias in our estimation, the lesser the variance. Similarly, less variance is often accompanied by more bias. Flexible(i.e. complex) models tend to be unbiased, but highly variable. Simple models are often extremely biased, but have low variance.

So for us, to select a model that appropriately balances the tradeoff between bias and variance, and thus minimizes the reducible error, we need to select a model of the appropriate flexibility for the data. However, this selected model should not overfit the data as well which we will discuss in the next chapter. Read [1](https://threadreaderapp.com/thread/1584515105374339073.html) and [2](https://www.simplilearn.com/tutorials/machine-learning-tutorial/bias-and-variance)





## Simulated Breakdown of the MSPE
 
Although the variance-bias trade-off conceptually seems intuitive, at least from a mathematical standpoint, another practical question arises: Is it possible to observe the components of the decomposed Mean Squared Prediction Error (MSPE)? In real-world data, observing these components is challenging as we do not know the true function and irreducible error. However, we can illustrate this concept through simulations.

For our analysis, we revisit the example discussed earlier. We consider years of schooling, varying between 9 and 16 years, as our variable of interest. From this 'population', we repeatedly take random samples. The objective now is to utilize each sample to develop a predictor or a set of prediction rules. These rules aim to predict a number or a series of numbers (years of schooling in this simulation) that are also drawn from the same population. By doing so, we can effectively simulate and analyze the decomposition of the MSPE, providing a clearer understanding of its components and their interplay.


```r
# Here is our population
populationX <- c(9,10,11,12,13,14,15,16)


#Let's have a containers to have repeated samples (2000)
Ms <- 5000
samples <- matrix(0, Ms, 10)
colnames(samples) <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10")

# Let's have samples (with replacement always)
set.seed(123)
for (i in 1:nrow(samples)) {
  samples[i,] <- sample(populationX, 10, replace = TRUE)
}
head(samples)
```

```
##      X1 X2 X3 X4 X5 X6 X7 X8 X9 X10
## [1,] 15 15 11 14 11 10 10 14 11  13
## [2,] 12 14 14  9 10 11 16 13 11  11
## [3,]  9 12  9  9 13 11 16 10 15  10
## [4,]  9 14 11 12 14  9 11 15 13  12
## [5,] 15 16 10 13 15  9  9 10 15  11
## [6,] 12 13 15 13 11 16 14  9 10  13
```

Now, Let's use our predictors: 



```r
# Container to record all predictions
predictions <- matrix(0, Ms, 2)

# fhat_1 = 10
for (i in 1:Ms) {
  predictions[i,1] <- 10
}

# fhat_2 - mean
for (i in 1:Ms) {
  predictions[i,2] <- sum(samples[i,])/length(samples[i,])
}

head(predictions)
```

```
##      [,1] [,2]
## [1,]   10 12.4
## [2,]   10 12.1
## [3,]   10 11.4
## [4,]   10 12.0
## [5,]   10 12.3
## [6,]   10 12.6
```

Now let's have our MSPE decomposition:


```r
# MSPE
MSPE <- matrix(0, Ms, 2)
for (i in 1:Ms) {
  MSPE[i,1] <- mean((populationX-predictions[i,1])^2)
  MSPE[i,2] <- mean((populationX-predictions[i,2])^2)
}
head(MSPE)
```

```
##      [,1] [,2]
## [1,] 11.5 5.26
## [2,] 11.5 5.41
## [3,] 11.5 6.46
## [4,] 11.5 5.50
## [5,] 11.5 5.29
## [6,] 11.5 5.26
```

```r
# Bias
bias1 <- mean(populationX)-mean(predictions[,1])
bias2 <- mean(populationX)-mean(predictions[,2])

# Variance (predictor)
var1 <- var(predictions[,1])
var1
```

```
## [1] 0
```

```r
var2 <- var(predictions[,2])
var2
```

```
## [1] 0.5385286
```

```r
# Variance (epsilon)
var_eps <- mean((populationX-12.5)^2)
var_eps
```

```
## [1] 5.25
```

Let's put them in a table:


```r
VBtradeoff <- matrix(0, 2, 4)
rownames(VBtradeoff) <- c("fhat_1", "fhat_2")
colnames(VBtradeoff) <- c("Bias", "Var(fhat)", "Var(eps)", "MSPE")
VBtradeoff[1,1] <- bias1^2
VBtradeoff[2,1] <- bias2^2
VBtradeoff[1,2] <- var1
VBtradeoff[2,2] <- var2
VBtradeoff[1,3] <- var_eps
VBtradeoff[2,3] <- var_eps
VBtradeoff[1,4] <- mean(MSPE[,1])
VBtradeoff[2,4] <- mean(MSPE[,2])
round(VBtradeoff, 3)
```

```
##        Bias Var(fhat) Var(eps)   MSPE
## fhat_1 6.25     0.000     5.25 11.500
## fhat_2 0.00     0.539     5.25  5.788
```
  
This table clearly shows the decomposition of MSPE. The first column is the contribution to the MSPE from the bias, and the second column is the contribution from the variance of the predictor. These together make up the reducible error. The third column is the variance that comes from the data, the irreducible error. The last column is, of course, the total MSPE, and we can see that $\hat{f}_2$ is the better predictor because of its lower MSPE.  

Here is an example that shows the trade-off between bias and variance using prediction functions in a simulation.  

In a seminal study, Angrist and Krueger (1991) tackled the problems associated with ability bias in Mincer's equation by creating an instrumental variable based on students' quarter of birth. The data is provided as "ak91" in the package "masteringmetrics". Instead of using actual data, we simulate only education and income data using the same method of moments. One purpose is to show that we may perform simulations using only descriptive tables and regression results even if we do not have the actual data, allowing us to discuss the underlying concepts effectively.

In the simulation below, we generate a dataset based on the descriptive statistics of years of schooling (s) and the logarithm of wage (lnw). We then create four different prediction functions to evaluate their MSPE, bias, and variance.



```r
# Set seed for reproducibility
set.seed(123)

# Parameters
n <- 26732
mean_lnw <- 5.9
sd_lnw <- 0.7
mean_s <- 12.8
sd_s <- 3.3
min_s <- 0
max_s <- 20

# Generate schooling data
#schooling <- runif(n, min = min_s, max = max_s)
  # Generate schooling data and round to nearest 0.5
  schooling <- round(runif(n, min = min_s, max = max_s) * 2) / 2

# Generate lnw data using the provided regression coefficients
# Intercept = 5, Coefficient of s = 0.07
lnw <- 5 + 0.07 * schooling + rnorm(n, mean = 0, sd = sd_lnw)

# Create data frame
data <- data.frame(schooling = schooling, lnw = lnw)

# Define prediction functions for log income
predictor1 <- rep(mean(lnw), n)  # Constant predictor based on the mean of lnw
predictor2 <- 5 + 0.10 * schooling + rnorm(n, mean = 0, sd = 0.1)  # Linear predictor with noise
predictor3 <- 4.5 + 0.06 * schooling  # Biased predictor with lower intercept and coefficient
predictor4 <- 5 + 0.07 * schooling  # True predictor (linear)

# Function to calculate MSPE components
calculate_mspe <- function(true, pred) {
  bias <- mean(pred - true)
  variance <- var(pred)
  irreducible_error <- var(true - mean(true))
  mspe <- mean((true - pred)^2)
  return(c(bias^2, variance, irreducible_error, mspe))
}

# Calculate MSPE components for each predictor
true_values <- lnw

mspe1 <- calculate_mspe(true_values, predictor1)
mspe2 <- calculate_mspe(true_values, predictor2)
mspe3 <- calculate_mspe(true_values, predictor3)
mspe4 <- calculate_mspe(true_values, predictor4)

# Combine results into a data frame
results <- data.frame(
  Predictor = c("Predictor 1", "Predictor 2", "Predictor 3", "Predictor 4"),
  Bias2 = c(mspe1[1], mspe2[1], mspe3[1], mspe4[1]),
  Variance = c(mspe1[2], mspe2[2], mspe3[2], mspe4[2]),
  IrreducibleError = c(mspe1[3], mspe2[3], mspe3[3], mspe4[3]),
  MSPE = c(mspe1[4], mspe2[4], mspe3[4], mspe4[4])
)

# Print results with 3 decimal points
results_rounded <- data.frame(
  Predictor = results$Predictor,
  Bias2 = round(results$Bias2, 3),
  Variance = round(results$Variance, 3),
  IrreducibleError = round(results$IrreducibleError, 3),
  MSPE = round(results$MSPE, 3)
)

print(results_rounded)
```

```
##     Predictor Bias2 Variance IrreducibleError  MSPE
## 1 Predictor 1 0.000    0.000            0.661 0.661
## 2 Predictor 2 0.094    0.342            0.661 0.620
## 3 Predictor 3 0.353    0.120            0.661 0.847
## 4 Predictor 4 0.000    0.163            0.661 0.490
```

```r
# Clear the environment
rm(list = ls())
```

The simulation demonstrates the performance of four different prediction functions. Among these, the true linear predictor (Predictor 4) achieved the smallest MSPE, indicating it provides the most accurate predictions. This is expected since it closely aligns with the underlying data generation process.


```r
# Load necessary library
library(ggplot2)
library(gridExtra)  # For arranging plots side by side

# Set seed for reproducibility
set.seed(123)

# Parameters
n <- 26732
mean_lnw <- 5.9
sd_lnw <- 0.7
mean_s <- 12.8
sd_s <- 3.3
min_s <- 0
max_s <- 20

# Initialize storage for beta0 and beta1
beta0 <- numeric(100)
beta1 <- numeric(100)
mspe_values <- numeric(100)
bias_values <- numeric(100)
variance_values <- numeric(100)
irreducible_error_values <- numeric(100)

# Function to calculate MSPE components
calculate_mspe <- function(true, pred) {
  bias <- mean(pred - true)
  variance <- var(pred)
  irreducible_error <- var(true - mean(true))
  mspe <- mean((true - pred)^2)
  return(c(bias^2, variance, irreducible_error, mspe))
}

# Generate 100 different datasets and fit models
for (i in 1:100) {
  # Generate schooling data and round to nearest 0.5
  schooling <- round(runif(n, min = min_s, max = max_s) * 2) / 2
  
  # Generate lnw data using the provided regression coefficients
  lnw <- 5 + 0.07 * schooling + rnorm(n, mean = 0, sd = sd_lnw)
  
  # Fit the linear model
  model <- lm(lnw ~ schooling)
  beta0[i] <- coef(model)[1]
  beta1[i] <- coef(model)[2]
  
  # Generate predictions using the model
  predictions <- predict(model, newdata = data.frame(schooling = schooling))
  
  # Calculate MSPE components
  mspe_components <- calculate_mspe(lnw, predictions)
  mspe_values[i] <- mspe_components[4]
  bias_values[i] <- mspe_components[1]
  variance_values[i] <- mspe_components[2]
  irreducible_error_values[i] <- mspe_components[3]
  
  # Print initial 50 observations and other calculated variables for the first dataset
  if (i == 1) {
    initial_data <- data.frame(schooling = schooling[1:50], lnw = lnw[1:50], predicted_lnw = predictions[1:50])
    print("Initial 50 observations of schooling, lnw, and predicted lnw:")
    print(initial_data)
    
    print("Coefficients for the first generated dataset:")
    print(coef(model))
    
    print("MSPE components for the first generated dataset:")
    print(mspe_components)
  }
}
```

```
## [1] "Initial 50 observations of schooling, lnw, and predicted lnw:"
##    schooling      lnw predicted_lnw
## 1        6.0 4.309461      5.407463
## 2       16.0 5.252208      6.124208
## 3        8.0 5.800693      5.550812
## 4       17.5 6.868000      6.231720
## 5       19.0 6.337707      6.339232
## 6        1.0 5.271255      5.049090
## 7       10.5 6.936026      5.729998
## 8       18.0 6.449974      6.267557
## 9       11.0 6.726069      5.765835
## 10       9.0 5.014647      5.622486
## 11      19.0 6.781244      6.339232
## 12       9.0 4.850522      5.622486
## 13      13.5 6.106988      5.945022
## 14      11.5 5.621766      5.801673
## 15       2.0 4.888290      5.120764
## 16      18.0 6.967818      6.267557
## 17       5.0 5.339867      5.335788
## 18       1.0 4.622575      5.049090
## 19       6.5 6.853711      5.443300
## 20      19.0 7.802680      6.339232
## 21      18.0 5.595551      6.267557
## 22      14.0 6.682397      5.980859
## 23      13.0 6.445602      5.909184
## 24      20.0 6.644234      6.410906
## 25      13.0 6.194610      5.909184
## 26      14.0 6.800737      5.980859
## 27      11.0 6.830699      5.765835
## 28      12.0 4.844341      5.837510
## 29       6.0 5.324725      5.407463
## 30       3.0 5.177032      5.192439
## 31      19.5 7.673262      6.375069
## 32      18.0 6.750589      6.267557
## 33      14.0 5.300752      5.980859
## 34      16.0 6.542941      6.124208
## 35       0.5 3.807720      5.013253
## 36       9.5 6.065694      5.658324
## 37      15.0 6.444348      6.052534
## 38       4.5 6.583272      5.299951
## 39       6.5 5.447284      5.443300
## 40       4.5 5.759566      5.299951
## 41       3.0 5.401111      5.192439
## 42       8.5 5.123768      5.586649
## 43       8.5 5.225697      5.586649
## 44       7.5 4.734100      5.514974
## 45       3.0 5.616073      5.192439
## 46       3.0 5.388677      5.192439
## 47       4.5 4.772368      5.299951
## 48       9.5 4.891484      5.658324
## 49       5.5 4.781807      5.371625
## 50      17.0 5.894623      6.195883
## [1] "Coefficients for the first generated dataset:"
## (Intercept)   schooling 
##  4.97741540  0.07167454 
## [1] "MSPE components for the first generated dataset:"
## [1] 1.608257e-29 1.709228e-01 6.609322e-01 4.899911e-01
```

```r
# Calculate expected MSPE, bias, and variance
expected_mspe <- mean(mspe_values)
expected_bias <- mean(bias_values)
expected_variance <- mean(variance_values)
expected_irreducible_error <- mean(irreducible_error_values)

# Print results
cat("Expected MSPE:", round(expected_mspe, 3), "\n")
```

```
## Expected MSPE: 0.491
```

```r
cat("Expected Bias:", round(expected_bias, 3), "\n")
```

```
## Expected Bias: 0
```

```r
cat("Expected Variance:", round(expected_variance, 3), "\n")
```

```
## Expected Variance: 0.163
```

```r
cat("Expected Irreducible Error:", round(expected_irreducible_error, 3), "\n")
```

```
## Expected Irreducible Error: 0.654
```

```r
# Clear the environment
rm(list = ls())
```


Other simulation examples are [1](https://blog.zenggyu.com/en/post/2018-03-11/understanding-the-bias-variance-decomposition-with-a-simulated-experiment/) , [2](https://www.r-bloggers.com/2019/06/simulating-the-bias-variance-tradeoff-in-r/), and [3](https://daviddalpiaz.github.io/r4sl/simulating-the-biasvariance-tradeoff.html)

## Biased estimator as a predictor

Up to this point, we have shown through simulation that a prediction function with zero bias but high variance often produces better predictions than a prediction function with zero variance but high bias. However, we can potentially obtain an even better prediction function that has some bias and some variance. A better prediction function means a smaller Mean Squared Prediction Error (MSPE). The key idea is that if the reduction in variance more than compensates for the increase in bias, then we have a better predictor.

To explore this, let's define a biased estimator of \(\mu_x\):
\[
\hat{X}_{\text{biased}} = \hat{\mu}_x = \alpha \bar{X}
\]
where \(\bar{X}\) is the sample mean. The sample mean \(\bar{X}\) is an unbiased estimator of \(\mu_x\), and the parameter \(\alpha\) introduces bias. When \(\alpha\) is 1, \(\hat{\mu}_x\) becomes the unbiased sample mean.

The bias of the estimator \(\hat{\mu}_x\) is given by:
\[
\operatorname{Bias}(\hat{\mu}_x) = \mathbb{E}[\hat{\mu}_x] - \mu_x = \alpha \mu_x - \mu_x = (\alpha - 1) \mu_x
\]
The variance of the estimator \(\hat{\mu}_x\) is:
\[
\operatorname{Var}(\hat{\mu}_x) = \operatorname{Var}(\alpha \bar{X}) = \alpha^2 \operatorname{Var}(\bar{X})
\]
Since the variance of the sample mean \(\bar{X}\) is \(\frac{\sigma_{\varepsilon}^2}{n}\) (Check chapter 5.4), we have:
\[
\operatorname{Var}(\hat{\mu}_x) = \alpha^2 \frac{\sigma_{\varepsilon}^2}{n}
\]

The MSPE and its bias-variance components:
\[
\text{MSPE} = \mathbb{E}[(\hat{\mu}_x - \mu_x)^2] = \operatorname{Bias}^2(\hat{\mu}_x) + \operatorname{Var}(\hat{\mu}_x) + \sigma_{\varepsilon}^2
\]
where \(\sigma_{\varepsilon}^2\) is the irreducible error.

First, calculate the bias squared:
\[
\operatorname{Bias}^2(\hat{\mu}_x) = [(\alpha - 1) \mu_x]^2 = (1 - \alpha)^2 \mu_x^2
\]

Next, calculate the variance:
\[
\operatorname{Var}(\hat{\mu}_x) = \alpha^2 \frac{\sigma_{\varepsilon}^2}{n}
\]

Finally, the irreducible error is  \(\sigma_{\varepsilon}^2\). After combining these terms,we can shows that the MSPE for the biased estimator \(\hat{\mu}_x = \alpha \bar{X}\) is:
\[
\text{MSPE} = (1 - \alpha)^2 \mu_x^2 + \alpha^2 \frac{\sigma_{\varepsilon}^2}{n} + \sigma_{\varepsilon}^2 = [(1-\alpha) \mu_x]^2 + \frac{1}{n} \alpha^2 \sigma_{\varepsilon}^2 + \sigma_{\varepsilon}^2
\]

The final expression for the MSPE of the biased estimator \(\hat{\mu}_x = \alpha \bar{X}\) combines the squared bias term \((1 - \alpha)^2 \mu_x^2\), the variance term \(\frac{\alpha^2 \sigma_{\varepsilon}^2}{n}\), and the irreducible error term \(\sigma_{\varepsilon}^2\). By adjusting \(\alpha\), we can balance the trade-off between bias and variance to minimize the MSPE. This highlights that a small amount of bias can be beneficial if it significantly reduces the variance, leading to a lower MSPE and thus a better predictor.

Our first observation is that when $\alpha$ is one, the bias will be zero.  Since it seems that MSPE is a convex function of $\alpha$, we can search for $\alpha$ that minimizes MSPE.  The value of \(\alpha\) that minimizes the MSPE is:

$$
\frac{\partial \text{MSPE}}{\partial \alpha} =0 \rightarrow ~~ \alpha = \frac{\mu^2_x}{\mu^2_x+\sigma^2_\varepsilon/n}<1
$$

Check end of the chapter for step-by-step derivation of the optimal value of \(\alpha\) that minimizes the MSPE.

Using the same simulation sample above , lets calculate alpha and MSPE with this new biased prediction function, and compare all 3 MSPEs. 







```r
pred <-rep(0, Ms)

# The magnitude of bias
alpha <- (mean(populationX))^2/((mean(populationX)^2+var_eps/10))
alpha
```

```
## [1] 0.9966513
```

```r
# Biased predictor
for (i in 1:Ms) {
  pred[i] <- alpha*predictions[i,2]
}
# Check if E(alpha*Xbar) = alpha*mu_x
mean(pred)
```

```
## [1] 12.45708
```

```r
alpha*mean(populationX)
```

```
## [1] 12.45814
```

```r
# MSPE
MSPE_biased <- rep(0, Ms)
for (i in 1:Ms) {
  MSPE_biased[i] <- mean((populationX-pred[i])^2)
}
mean(MSPE_biased)
```

```
## [1] 5.786663
```
  
Let's add this predictor into our table:  


```r
VBtradeoff <- matrix(0, 3, 4)
rownames(VBtradeoff) <- c("fhat_1", "fhat_2", "fhat_3")
colnames(VBtradeoff) <- c("Bias", "Var(fhat)", "Var(eps)", "MSPE")
VBtradeoff[1,1] <- bias1^2
VBtradeoff[2,1] <- bias2^2
VBtradeoff[3,1] <- (mean(populationX)-mean(pred))^2
VBtradeoff[1,2] <- var1
VBtradeoff[2,2] <- var2
VBtradeoff[3,2] <- var(pred)
VBtradeoff[1,3] <- var_eps
VBtradeoff[2,3] <- var_eps
VBtradeoff[3,3] <- var_eps
VBtradeoff[1,4] <- mean(MSPE[,1])
VBtradeoff[2,4] <- mean(MSPE[,2])
VBtradeoff[3,4] <- mean(MSPE_biased)
round(VBtradeoff, 3)
```

```
##         Bias Var(fhat) Var(eps)   MSPE
## fhat_1 6.250     0.000     5.25 11.500
## fhat_2 0.000     0.539     5.25  5.788
## fhat_3 0.002     0.535     5.25  5.787
```

As seen , increase in bias is lower than decrease in variance. The prediction function with some bias and variance is the **best prediction function** as it has the smallest MSPE. This example shows the difference between estimation and prediction for a simplest predictor, the mean of $X$.  

In the next chapter, we will explore overfitting and how it relates to and differs from the bias-variance tradeoff. 

**Note:** To find the value of \(\alpha\) that minimizes the Mean Squared Prediction Error (MSPE), we take the derivative of the MSPE with respect to \(\alpha\), set it to zero, and solve for \(\alpha\).

The MSPE is given by:
\[
\text{MSPE} = [(1-\alpha) \mu_x]^2 + \frac{1}{n} \alpha^2 \sigma_{\varepsilon}^2 + \sigma_{\varepsilon}^2
\]

First, we take the derivative of MSPE with respect to \(\alpha\):
\[
\frac{\partial \text{MSPE}}{\partial \alpha} = \frac{\partial}{\partial \alpha} \left[(1-\alpha)^2 \mu_x^2 + \frac{1}{n} \alpha^2 \sigma_{\varepsilon}^2 + \sigma_{\varepsilon}^2\right]
\]

We compute the derivative term by term:
\[
\frac{\partial \text{MSPE}}{\partial \alpha} = 2(1-\alpha)(-1) \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2
\]
Simplifying, we get:
\[
\frac{\partial \text{MSPE}}{\partial \alpha} = -2(1-\alpha) \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2
\]

To find the minimum MSPE, we set the derivative equal to zero:
\[
-2(1-\alpha) \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2 = 0
\]

Solving for \(\alpha\):
\[
-2 \mu_x^2 + 2 \alpha \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2 = 0
\]
\[
- \mu_x^2 + \alpha \mu_x^2 + \frac{1}{n} \alpha \sigma_{\varepsilon}^2 = 0
\]
\[
- \mu_x^2 + \alpha \left( \mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n} \right) = 0
\]
\[
\alpha \left( \mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n} \right) = \mu_x^2
\]
\[
\alpha = \frac{\mu_x^2}{\mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n}}
\]

Thus, the value of \(\alpha\) that minimizes the MSPE is:
\[
\alpha = \frac{\mu_x^2}{\mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n}}
\]




