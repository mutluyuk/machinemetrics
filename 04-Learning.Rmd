
# Learning 

Imagine a young and curious person named Alex who occasionally gets sick after eating but doesn't know which specific food is the culprit. He grappled with recurring allergic reactions. Driven by a pressing need to pinpoint their root cause, he embarked on a personal mission. Eager to understand the extent of his susceptibility, Alex sought patterns and clues.

Data Collection: Each time Alex eats, he makes a list of everything he's consumed that day. He also records how he felt afterward, specifically noting any days he felt ill.

Pattern Recognition: After several weeks, Alex starts to see a pattern. Every time he ate dishes containing garlic, he felt sick within a few hours. However, on days he avoided garlic, he generally felt fine.

Making Predictions: With this new insight, Alex hypothesizes that garlic might be the cause of his discomfort. To test this theory, he deliberately avoids garlic for a few days and notes his health. Conversely, on another day, with all other foods being consistent, he consumes garlic to see if the reaction recurs.

Validation: During the days without garlic, Alex feels perfectly fine. However, after the day he reintroduces garlic, the familiar sickness returns. This strengthens Alex's belief in the connection between garlic and his adverse reactions.

Updating the Model: Wanting to be thorough, Alex decides to test other ingredients, wondering if the allergy might extend beyond garlic. After trying onions and shallots on different days and noticing no adverse reactions, Alex concludes that his allergy seems specific to garlic.

In this example, Alex is acting like a basic machine learning model.

Collect data (observations of the food).

Recognize patterns in the data.

Make informed predictions based on identified patterns.

Validate predictions against actual occurrences.

Adjust the predictive model considering new or contradictory data.

While Alex's learning process strongly suggests that garlic triggers his symptoms, it's important to recognize the limitations of his informal "model." Just as in any learning model, prediction errors could arise from multiple factors. For instance, there might be times when he consumes garlic but doesn't get sick because of variations in the quantity consumed, or the form in which it's ingested (raw versus cooked). There could also be external factors, like the combination of garlic with other foods, that influence his reaction. It's also possible that, on some days, other confounding variables like stress or a different underlying illness might mask or exaggerate his garlic-induced symptoms. Thus, while Alex feels confident in his findings, he understands that real-life scenarios can introduce unpredictability, making it essential to continually refine and reassess his conclusions.

This analogy showcases the parallels between Alex's learning process and that of machines. In a manner mirroring machine learning's approach to refining predictions with data, Alex has learned his very rare allergy. While he might not grasp the precise mechanisms or reasons behind his garlic allergy, and there could be some prediction errors, he has effectively employed a method reminiscent of machine learning. The insights he's gained are invaluable, allowing him to make informed decisions about his diet in the future.



Machine Learning comes more from the computer science field. While machine learning also wants to understand and interpret data similar to statistical learning as we talked in previous chapter, its main goal often leans more towards making accurate predictions or decisions. It might not always need to understand why something happens as long as it can predict it accurately. Think of it as an engineer who builds a tool that works efficiently, even if they don't fully grasp the science behind every component.

So, why do some people call machine learning "statistical learning"? Because many machine learning techniques are grounded in statistical methods. As both fields evolved, they started to overlap more and more. In many cases, the boundary between them has become blurry. For a layperson, you might think of statistical learning as a close cousin to machine learning. They both aim to understand and learn from data, but they might have slightly different priorities or approaches. At its heart, machine learning is a method of teaching computers to make decisions or predictions based on data, rather than being explicitly programmed to do so.

In machine learning, the main aim is to create a model with specific settings or parameters. This model should be good at making predictions on new, unseen data. Later in this book, we'll explore the three key steps in machine learning: prediction, training, and hyperparameter tuning.

Everything in machine learning starts with data. This could be photos, texts, economic indicators, political polling data, public health records, or employment statistics. This data is then organized and used as training data for the machine learning model. Generally, the more data we have, the better the model will be.

Once we have the data ready, the next steps are clear:

First, there's the Prediction or Inference Phase. This stage unfolds when a predictor, already trained, is employed on test data it hasn't encountered before. By this time, the model and its settings are already decided, and we're just using it to make predictions on new data points.

Next is the Training or Parameter Estimation Phase. This is when we refine our predictive model using training data. Broadly, there are two strategies to pinpoint robust predictors given our data. One strategy is searching for the best predictor based on a designated measure of quality, often referred to as identifying a point estimate. The other is Bayesian inference outscope of this book. Irrespective of the strategy, the goal remains the same: using numerical methods to find parameters that align or "fit" our data. 

Last, we have the Hyperparameter Tuning or Model Selection Phase. Here, we're trying to pick the optimal model and its corresponding hyperparameters on how well they do on a test or validation data. The main goal is to find a model that not only works well on our training data but will also make good predictions on new data it hasn't seen before.

After gathering and preparing the data, we select a model, train it, evaluate its performance, and fine-tune its parameters. Once optimized, the model is then employed to make predictions. In this section, we've touched upon the concepts of prediction, training, and hyperparameter tuning. However, rest assured, in future chapters, we will delve into these topics in comprehensive detail.

Learning systems in machine learning refer to algorithms and models that improve their performance or adapt to new data over time without being explicitly programmed for each specific task. These systems "learn" from data, identifying patterns or making decisions based on input.

In the realm of machine learning, systems are designed for various purposes and roles. Together, these roles underscore what machine learning models strive to accomplish, each offering insights and recommendations suited to particular situations and requirements.

The Descriptive role is foundational. True to its name, it's centered around understanding and articulating the information encapsulated within the data. This is particularly useful for understanding historical trends and patterns.

Next is the Predictive role. This goes a step beyond by not just reflecting on past events but by forecasting the future one. By analyzing existing data, the system forms informed predictions about upcoming occurrences.

Lastly, the Prescriptive role stands out. It's arguably the most forward-looking of the trio. Rather than merely explaining or projecting, it escalates its function by recommending specific courses of action derived from its data analysis. For instance, industrial economists using machine learning might recommend optimal pricing strategies, suggest resource allocation, and guide firms on ideal hiring practices by analyzing consumer demand, competitor prices, product line performance, employment trends, and skill set data. Moreover, based on diverse data such as patient outcomes, resource availability, population health trends, drug efficacy, and claim histories, health economists can use machine learning to recommend cost-effective treatment paths, optimize resource allocation in hospitals, suggest preventive interventions for high-risk groups, advise on pharmaceutical pricing, and offer insights into health insurance premium setting.

In the next subsection, we'll explore step by step how to find and train models, and how these models generalize well to unseen data.


## Learning Systems {-}

A Machine Learning System is a dynamic framework often seen in various Machine Learning projects. This system generally unfolds in an iterative manner, encompassing several key phases. It all begins with Data Collection, where datasets are not only created but also maintained and updated. This phase is crucial as the quality and relevance of data significantly influence the outcomes. Following this, the system dives into the Experimentation phase. During this stage, the data is thoroughly explored, various hypotheses about the data and potential models are formulated, tested, and validated, resulting in the construction of both training and prediction pipelines. With a robust model at hand, the next step is Deployment, which entails integrating this model into a tangible, working product. But the journey doesn't conclude here. The Operations phase ensures that the deployed model is under constant surveillance, ensuring it remains current and in tune with the ever-evolving environment. Together, these phases epitomize the core mechanics of a Machine Learning System, emphasizing continuous learning and adaptation.

Machine learning systems are algorithms and models that adapt to new data over time, improving their performance without specific programming for every task. These systems "learn" from data, identifying patterns or making decisions based on given input. In this book, our primary focus will be on machine learning models, which are integral components of learning systems. A machine learning model combines programming code with data. How do we find, train, and make sure our machine learning models work well on unseen data? 

Here's a step-by-step guide:

Splitting the Data: We take our whole dataset and split it into two parts: one for training our model, training data, and one for testing its performance, testing data.

Use Training Data Wisely: Most of the time, we use 80% of our total data for training. But even within this training data, we divide it further: estimation data and validation data.

Keeping Testing Data Untouched: We don't use the testing data to make any decision that lead to the selection of the model. It's set aside only to see how well our finished model works.

Choose Possible Models: Before we start training, we decide on a few potential models we might want to use. For parametric models: We assume a form of the model up to but not including the model parameters. For nonparametric models: We pick feature variables to be used and possible values of any tuning parameters.

Training the Model: We take each potential model and train(fit) it using the estimation data which is usually bigger chunk of our training data.

Check and Compare Models: Once trained, we see how well each model does using validation data, usually the smaller chunk of training data it hasn't seen before. This performance is all about how good the model is at predicting on the validation data which was not used to train the models.

Pick the Best Model: Based on how well they did on the validation performance, we choose the best model.

Final Training: With our best model in hand, we then use all the training data, both estimation and validation data together, to give it one last thorough training.

Test: Finally, we use our untouched test data to estimate model performance and see how well the best model does.

Note that we are using the validation data to select a model, while the testing data is used to estimate model performance.

Above, we outlined the machine learning process using plain language for clarity. Below, we'll delve into these steps using more technical terms commonly accepted in the field. If certain concepts seem unfamiliar, rest assured, we'll unpack each one in greater detail in the upcoming chapters.

1. The learner has a sample of observations.  This is an arbitrary (random) set of objects or instances each of which has a set of features ($\mathbf{X}$ - features vector) and labels/outcomes ($y$).  We call this sequence of pairs as a training set: $S=\left(\left(\mathbf{X}_{1}, y_{1}\right) \ldots\left(\mathbf{X}_{m}, y_{m}\right)\right)$.  
2. We ask the learner to produce a **prediction rule** (a predictor or a classifier model), so that we can use it to predict the outcome of **new** domain points (observations/instances).  
3. We assume that the training dataset $S$ is generated by a data-generating model (DGM) or some "correct" labeling function, $f(x)$.  The learner does not know about $f(x)$.  In fact, we ask the learner to discover it.  
4. The learner will come up with a **prediction rule**, $\hat{f}(x)$, by using $S$, which will be different than $f(x)$.  Hence, we can measure the learning system's performance by a loss function:  $L_{(S, f)}(\hat{f})$, which is a kind of function that defines the difference between $\hat{f}(x)$ and $f(x)$. This is also called as the **generalization error** or the **risk**.  
5. The goal of the algorithm is to find $\hat{f}(x)$ that minimizes the error with respect to the unknown $f(x)$. The key point here is that, since the learner does not know $f(x)$, it cannot calculate **the loss function**.  However, it calculates the training error also called as the **empirical error** or the **empirical risk**, which is a function that defines the difference between $\hat{f}(x)$ and $y_i$.  
6. Hence, the learning process can be defined as coming up with a predictor $\hat{f}(x)$ that minimizes the empirical error.  This process is called **Empirical Risk Minimization** (ERM).  
7. Now the question becomes what sort of conditions would lead to bad or good ERM?  

If we use the training data (in-sample data points) to minimize the empirical risk, the process can lead to $L_{(S, f)}(\hat{f}) = 0$.  This problem is called **overfitting** and the only way to rectify it is to restrict the number of features in the learning model.  The common way to do this is to "train" the model over a subsection of the data ("seen" or in-sample data points) and apply ERM by using the test data ("unseen" or out-sample data points).  Since this process restrict the learning model by limiting the number of features in it, this procedure is also called **inductive bias** in the process of learning.  

There are always two "universes" in a statistical analysis: the population and the sample.  The population is usually unknown or inaccessible to us. We consider the sample as a random subset of the population.  Whatever the statistical analysis we apply almost always uses that sample dataset, which could be very large or very small.  Although the sample we have is randomly drawn from the population, it may not always be representative of the population.  There is always some risk that the sampled data happens to be very unrepresentative of the population.  Intuitively, the sample is a window through which we have partial information about the population.  We use the sample to **estimate** an unknown parameter of the population, which is the main task of **inferential statistics**.  Or, we use the sample to develop a **prediction rule** to predict unknown population outcomes.  

When we have a numeric outcome (non-binary), the lost function, which can be expressed as the **mean squared error** (MSE), assesses the quality of a **predictor** or an **estimator**. Note that we call $\hat{f}(x)$ as a predictor or estimator.  Can we use an **estimator** as a **predictor**?  Could a "good" estimator also be a "good" predictor.  We had some simulations in the previous chapter showing that the best estimator could be the worst predictor.  Why? In this section we will try to delve deeper into these questions to find answers.  

The starting point will be to define these two different but similar processes. 

