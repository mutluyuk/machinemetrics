
# Counterfactual:

Imagine a college student,Alex, who occasionally gets sick after eating but doesn't know which specific food is the culprit. In a community increasingly concerned about food allergies, this young individual grappled with recurring allergic reactions. Driven by a pressing need to pinpoint the root cause, he embarked on a personal mission. Eager to understand the extent of his susceptibility, Alex meticulously documented every instance in which he consumed food, noting all the ingredients, leading to a comprehensive data collection effort.

In his quest to uncover the reason for his sporadic sickness after eating, Alex adopted an approach that resembled common quantitative research methods: collecting data, utilizing descriptive statistics, visualizing the data, finding correlations, and ultimately using methods to determine the causation of the allergy.

As Alex continues to gather more data about his dietary intake and subsequent health reactions, he starts by creating a simple table for each day. After visualizing the data, he begins to spot correlations between certain foods and his well-being.

**Data Visualization**:

Based on this recorded data, Alex can employ various visualization methods to better understand and identify patterns:

a) For the Histogram of Reaction Intensity, this chart allows Alex to see the frequency of days with varying intensities of reactions. The X-axis represents the Reaction Intensity ranging from 1 to 10, while the Y-axis shows the Number of Days. An observation he might make is if he notices a high number of days with intensities around 8-10 after consuming garlic, providing an initial clue.

b) In the Bar Chart of Reactions by Food, he can visually compare the average reaction intensities for various foods. The X-axis displays different foods or ingredients such as garlic, dairy, and wheat. The Y-axis represents the Average Reaction Intensity. If the bar for garlic consistently stands out in comparison to other foods, it further signals a potential issue.

c) The Time Series Line Graph enables Alex to track the evolution of reactions over time. With the X-axis indicating the Date and the Y-axis highlighting the Reaction Intensity, a line tracing the intensity of reactions over time can help him pinpoint if certain clusters of high-intensity days align with the consumption of specific food.

Recognizing correlations visually with these tools means Alex can discern if there's a pronounced spike in the histogram every time garlic is consumed. The bar chart might indicate that garlic has a noticeably higher average reaction intensity than other foods. Similarly, the time series graph can demonstrate peaks in reaction intensities on specific dates, which Alex can then cross-reference with the food he consumed on those days.

By visually plotting the data, Alex can more effectively recognize patterns and correlations, offering a foundational understanding before venturing into more intricate statistical analyses.

**Delving Into Correlations**:

Observing Correlations: Alex began to rank his reactions on a scale from 1 to 10, with 10 marking the most severe reaction. As days turned to weeks, he noticed that every time he consumed garlic, the intensity of his reaction consistently hovered between 8 to 10. Contrastingly, other foods such as dairy or wheat might occasionally result in a reaction intensity of 3 or 4, but not always.

Potential Confounding Factors: On a particular day, Alex felt unwell after a meal without garlic, but recalled having a milkshake. Wondering if dairy might be another trigger, he started noting down dairy consumption alongside garlic. However, after several dairy-heavy days without any reaction, it becomes clear that the milkshake incident might have been a coincidence or caused by another factor.

Strength of Correlation: As weeks go by, the association between garlic ingestion and feeling under the weather becomes more evident. The consistency and strength of this correlation are much higher than with any other food. In statistical terms, one might say that garlic have a strong positive correlation with Alex's adverse reactions.

Spurious Correlations: A pattern Alex took note of was his increased tendency to fall ill on weekends. However, after some contemplation, he discerned that weekends were when he often dined out, inadvertently upping the odds of ingesting garlic. his is an example of a spurious correlation: the actual problem wasn't the weekend itself, but rather the increased exposure to the allergen.

Drawing Conclusions: While correlation does not imply causation, the consistent and strong correlation between garlic consumption and adverse reactions, gives Alex confidence in the hypothesis that he is allergic to garlic.

In this example, Alex's observations and data tracking are analogous to the process of determining correlation in statistical or machine learning contexts. Correlations can highlight patterns, yet it's crucial to ensure that confounding factors or spurious correlations aren't misleading the conclusions.

**The Mystery of Mia’s Illness:**

On certain weekends, Mia, Alex's girlfriend, also started feeling unwell. As she began to correlate her illness to the days she spent with Alex, she grew concerned. Was she allergic to something at Alex’s place? Or, even more alarmingly, was she developing an allergy to garlic, having shared many garlic-laden dishes with him?

Mia decided to chart her symptoms alongside Alex’s diary of garlic consumption. To her surprise, she found that she felt sick on several occasions when Alex had garlic in his meals, even if she hadn’t consumed any garlic herself.

Spurious Correlation Revealed:Further probing revealed an interesting detail. Whenever Alex prepared dishes with garlic at his place, he'd also light up a particular brand of aromatic candle to mask the strong garlic smell. Mia wasn't reacting to the garlic, but to the scent of that specific candle. Her sickness wasn’t directly linked to the days Alex consumed garlic, but rather to the days the candle was lit. The correlation between her sickness and Alex's garlic consumption was spurious, with the actual causative agent being the candle’s aroma.

In this example, Mia’s conclusion, based on initial observations, would lead her down the wrong path, emphasizing the importance of not mistaking correlation for causation. It serves as a cautionary tale on the pitfalls of spurious correlations in both real-life and statistical contexts.

**Alex's Deep Dive into the effect of his garlic consumption on his allergy severity:**

After discovering a strong correlation between his garlic consumption and allergic reactions, Alex decided to take his investigation a step further. While the correlation was evident, he wanted to quantitatively understand the exact impact of garlic consumption on his reactions. He suspected that while garlic was the primary association with his reactions, other variables might exacerbate or alleviate his symptoms. Beyond just the amount of garlic he consumed, could factors like his weight, the weather temperature, and even eating outside influence the severity of his reactions?

Gathering Data: For several weeks, Alex meticulously documented the amount of garlic in his meals, his weight each day, the day's peak weather temperature, whether he dined inside or outside.

To understand the relationship better, Alex used an Ordinary Least Squares (OLS) regression. This approach would allow him to understand how each variable, when considered together, might predict the severity of his allergic reaction. He find that the coefficient for garlic quantity was positive, reaffirming that the more garlic he consumed, the stronger the allergic reaction. Interestingly, on days when he weighed more, the severity of his allergic reaction was slightly less, all else being equal. Perhaps his body was better equipped to handle allergens when he was at a slightly higher weight. On warmer days, Alex's allergic reactions were milder than on colder days. Dining outside frequently correlated with more intense reactions. This puzzled Alex until he realized that eating outside often meant dining at restaurants or cafes where he had less control over ingredients, and the chance of consuming hidden garlic was higher.

Alex remembered that his girlfriend once mentioned he seemed to react more during weekends. Reflecting on it, he saw that weekends were indeed when they often dined out, leading to more exposure to garlic-rich dishes. It wasn't the fact that it was a weekend causing the reactions but the increased likelihood of eating garlic-containing food outside. This was a classic case of spurious correlation; the real culprit was the garlic, not the weekend!

Equipped with these insights, Alex made some lifestyle changes. He became cautious about eating out, especially on weekends. He also kept an eye on the day's temperature, preparing for potential reactions on colder days. Knowing that his weight had a buffering effect was an added insight, but he decided that a balanced diet and regular exercise were more crucial for his overall health.

**Investigating Causation**

Building on the previously identified correlation between garlic and adverse reactions, Alex feels the pressing need to ascertain whether garlic truly triggers his allergic responses. Although correlation had provided some preliminary insights, he recognized the limitations of correlation evidence in proving causation. He turned to Ordinary Least Squares (OLS) regression analysis, aiming to isolate the impact of garlic relative to other potential variables, like his weight, weather temperature, and the environment where he eats.

He remembered a recent news article discussing certain foods that were structurally and chemically similar to garlic. The article suggested that these foods could also trigger allergic reactions in individuals sensitive to garlic. This revelation complicated his inquiry, as neither correlation nor regression methods could offer him a definitive answer. Could there be other foods amplifying his reactions? Or was garlic the sole offender?

Determined to get to the bottom of this mystery, Alex decided to undertake a more rigorous approach: the experimental method. Often hailed as the gold standard for establishing causality, this method would allow Alex to control specific variables and thereby isolate the effects of garlic and other similar foods on his system. By methodically introducing and removing these foods from his diet in a controlled setting, he aimed to definitively ascertain the root cause of his adverse reactions.

To unravel this mystery, Alex approached his friend Mia, who didn't have any known food allergies, to participate in a controlled experiment. By having Mia as a control group, Alex could compare reactions between them, potentially teasing out the specific effects of garlic.

They both embarked on a week-long experiment, where their diets were standardized, with the only variance being the consumption of garlic and its similar foods. Mia's consistent lack of adverse reactions when consuming the same meals as Alex, especially those containing garlic, reinforced its role in Alex's allergic symptoms. Meanwhile, Alex's symptoms persisted, lending more weight to the hypothesis about garlic's culpability.

When Mia remained symptom-free even after consuming the foods similar to garlic that the news had warned about, it provided Alex with further clarity. It became evident that while those foods might be problematic for some, they weren't the culprits in Alex's case. By incorporating Mia into the experiment as a control group, Alex was not only able to more confidently ascertain the role of garlic in his reactions but also to rule out other potential allergens. 

Causation Established: With consistent results across multiple trials, combined with the knowledge that other potential causes have been ruled out, Alex concludes that garlic is not just correlated with, but is the actual cause of his allergic reactions. In scientific terms, Alex has moved from observing a correlation (a relationship between garlic consumption and allergic reactions) to establishing causation (determining that garlic directly causes the allergic reactions).

This journey mirrors the scientific process where controlled experiments, repeated trials, and the isolation of variables are crucial for determining the true cause of an observed effect.


## Qualitative and Quantitative research methods:

In the realm of research, there are two primary methodologies: qualitative and quantitative. Qualitative research methods often involve focus groups, unstructured or in-depth interviews, and the review of documents to discern specific themes. For instance, in social sciences, a qualitative study might explore the lived experiences of individuals living in poverty, capturing their stories and challenges through in-depth interviews. In economics, qualitative research might delve into understanding the socio-economic factors influencing a community's resistance to adopting digital currencies.

On the other hand, quantitative research typically employs surveys, structured interviews, and measurements. It also involves reviewing records or documents to gather numeric or quantifiable data. Quantitative methods emphasize objective measurements and the statistical, mathematical, or numerical analysis of data collected through polls, questionnaires, and surveys. An example from economics might be a study analyzing the correlation between unemployment rates and economic recessions using historical data. Another economic example could be a quantitative analysis of the impact of interest rate changes on consumer spending patterns over a decade.

Additionally, quantitative research can involve manipulating pre-existing statistical data using computational techniques. For instance, in economics, researchers might use computational models to predict the future growth rate of an economy based on various indicators. Quantitative research is not just a method but a way to learn about a specific group of people, known as a sample population. In the health sector, a quantitative study might examine the efficacy of a new drug on a sample population, measuring specific health outcomes and side effects. In economics, a study might evaluate the spending habits of millennials compared to baby boomers using structured surveys. Through scientific inquiry, it relies on data that are observed or measured to examine questions about this sample population. There are various designs under quantitative research, including Descriptive non-experimental, Quasi-experimental, and Experimental.

The processes that underpin these two research types differ significantly. Qualitative research is characterized by its inductive approach, which aids in the formulation of theories or hypotheses. In contrast, quantitative research follows a deductive approach, aiming to test predefined concepts, constructs, and hypotheses that together form a theory.

When considering the nature of the data, qualitative research is inherently subjective. It seeks to describe issues or conditions from the vantage point of those experiencing them. For example, in economics, a qualitative study might investigate the perceptions of small business owners towards global trade agreements. Quantitative research, however, is more objective. It focuses on observing the effects of a program on an issue or condition, with these observations subsequently interpreted by researchers.

The type of data these methodologies yield is also distinct. Qualitative research is text-based, delving deep to provide rich information on a limited number of cases. Quantitative research, meanwhile, is number-based, offering a broader scope of information but spread across a larger number of cases, often sacrificing depth for breadth.

In terms of response options, qualitative research tends to use unstructured or semi-structured options, allowing for more open-ended answers. Quantitative research, in contrast, relies on fixed response options, measurements, or observations. Furthermore, while qualitative research does not typically employ statistical tests in its analysis, quantitative research does, ensuring a more structured and numerical interpretation of data.

Lastly, when it comes to generalizability, qualitative research findings are often less generalizable due to their in-depth focus on specific cases. Quantitative research, with its broader scope, tends to be more generalizable to larger populations.

In summary, while both qualitative and quantitative research methodologies offer valuable insights, they differ in their methods, processes, nature of data, and generalizability, each serving unique purposes in the research landscape, as evidenced by their applications in fields like economics, social sciences, and health. In this book, we cover quantitative methods. 

https://libguides.usc.edu/writingguide/quantitative

## Quantitative - Research methods :

General perspective about research methods in health, economics, and social sciences:

A researcher asks a good answerable question. It does not mean we can always find the answer right away with available data and methods.  Yet we know that we can find an answer which will expand our knowledge of how the world works. A good answerable research question can be defined as a hypothesis which can be a phenomenon what we observe in the world. Also, hypothesis, that we want to prove, comes from theory as well.  If the theory explains or predicts this is the specific hypothesis how the world works, then we should observe it with the data. 

To answer these questions, we collect or obtain data, then explore data.  After making certain assumptions, we analyze the data. Then, we reach conclusions which can be associations, correlations, or causal relations. We use results to explain, extrapolate or predict! our hypothesis.	 We covered these concepts in detail in the introduction of this book.

Different fields have dominant methods within their field to answer research questions. 
[The main source for health section is C. Manski, Patient Care under Uncertainty, Princeton University Press, 2019.]
Health research use “Evidence Based Research!” 	
Manski told in his seminal book “Research on treatment response and personalized risk assessment shares a common objective: Probabilistic prediction of some patient outcome conditional on specified patient attributes...Econometricians and Statisticians refer to conditional prediction as regression, a term in use since the nineteenth century. Some psychologists use the term actuarial prediction and statistical prediction. Computer scientists may refer to machine learning and artificial intelligence. Researchers in business school may speak of predictive analytics.”
In most general way, after collecting and analyzing data, they present descriptive analysis seeks to understand associations. By various medical research methods, especially “Gold standard methods!”, when(if) they determine X causes Y. They propose treatment and surveillance. By using clinical trials, they want to determine treatment/surveillance and find its efficacy and effectiveness. Using Prescriptive analyses, they attempt to improve the performance of actual decision making. They try to find optimal solution between surveillance and aggressive treatment. Mainly, they use clinical judgment and evidence-based research.
In general, statistical imprecision and identification problems affect empirical (evidence-based) research that uses sample data to predict population outcomes. There is a tension between the strength of assumptions and their credibility. The credibility of the inference decreases with the strength of the assumptions maintained.

The most credible and acceptable method is The Gold Standard! In health.  The "Gold Standard" Method for Health researchers is obtaining the effect of tested treatment comparing the results from trials and experiments which has treatment and control groups, and. In machine learning this is known as A/B testing, in economics it is random and field experiments. 

Even though this method is the most credible method in empirical research, it has problems like any other empirical methods. First, study/trial sample and actual patient population can be very different. Second, Due to small sample sizes, estimates and identification of treatment effects are imprecise. Third, it is wishful extrapolation to assume that treatment response in trials performed on volunteers is the same as what would occur in actual patient populations. Thus, predictions are fragile as they have limited data and do not handle uncertainty sufficiently.

Most of health researcher give more value for the results obtained by a randomized trial with 200 observations than results from observational studies with 200,000 observations. Why do most of the medical and health researchers have this perspective?	 To justify trials performed on study populations that may differ substantially from patient populations, researchers in public health and the social sciences often cite Donald Campbell, who made a distinction between the internal and external validity of studies of the treatment response (Campbell and Stanley, 1963). A study has internal validity if it has credible findings for the study population (in-sample data in Machine Learning). It has external validity if an invariance assumption permits credible extrapolation (out-sample in ML). The appeal of randomized trials is their internal validity. Campbell argued that studies should be judged primarily by their internal validity and secondarily by their external validity. Since 1960s, this perspective has been used to argue for the primacy of experimental research over observational studies, whatever the study population may be.

In contrast, observational studies which uses the representative sample of the population have more credibility in economics than randomized control trials with a small sample.  The Campbell perspective has also been used to argue that the best observational studies are those that most closely approximate randomized experiments if they are done with representative samples. 

We should keep in mind that statistics and economics have the following perspective which is weird in other fields. For instance, "All models are wrong, but some are useful" (Box, 1976), and "Economists should not worry if their assumptions are wrong, as long as their conclusions are relatively accurate." (Friedman, 1953)

## Data and visualization

Data and Visualization in Health, Economics, Business, and Social Sciences

Researchers in fields such as health, economics, business, and social sciences primarily utilize three types of datasets: cross-sectional data, time series data, and panel data (also known as longitudinal data). 

Cross-sectional data is collected at a single point in time or over a short period from a sample of subjects, such as individuals, companies, or countries. This type of data is often used to describe the characteristics or attributes of a group at a specific point in time.

Time series data, collected over an extended period at regular intervals, tracks changes in a particular variable over time. This data is useful for identifying trends and patterns over time and making predictions about future developments.

Panel data, collected over an extended period from the same sample of subjects, tracks changes in a particular variable over time for each subject in the sample. This type of data is valuable for studying the relationship between different variables and identifying trends and patterns over time.

When working with any type of data, it is crucial to follow the following steps:

Examine the raw data: Understand the structure and content of the data.

Clean and prepare the data: Check for errors, missing values, and outliers, and ensure that the data is in a usable format.

Understand the data: Investigate the variables and their relationships through statistical analysis, visualizations, and other methods.

Effective visualization can reveal data features that inform further analysis. Visualizing data is an essential step in the process of understanding and analyzing data in various fields. Effective visualization techniques, such as histograms, barplots, boxplots, scatterplots, and heatmaps, can provide valuable insights into patterns and trends within the data, guiding further analysis and decision-making.

A histogram is a graphical representation that displays the frequency or distribution of a set of continuous or discrete data. It helps visualize the data and understand the underlying distribution. A histogram comprises a set of bins, which represent ranges of values, and the height of each bin indicates the frequency of data points within that range. For instance, a histogram displaying the weights of a group of animals might have bins representing weight ranges (e.g., 20-29 kg, 30-39 kg, etc.), and the height of each bin would indicate the number of animals within that weight range.

A barplot is a graphical representation that illustrates the mean or median of a dataset. It serves to visualize the central tendency of the data and compare different groups or categories. A barplot comprises a set of bars, with the height of each bar representing the mean or median of the data for that group or category. For instance, a barplot might compare the average fuel efficiency of various car models or the median home prices in different neighborhoods.

A boxplot is a graphical representation that displays the distribution of a dataset. It helps visualize the spread, skewness, and potential outliers in the data. A boxplot consists of a box representing the interquartile range (the middle 50% of the data), a line denoting the median, and "whiskers" extending from the box to the minimum and maximum values of the data. Boxplots are particularly useful for comparing the distributions of different groups or categories of data, such as the distribution of exam scores for students in different classes.

A scatterplot is a graphical representation that exhibits the relationship between two variables. It enables the visualization of the relationship between the variables and identification of patterns and trends. A scatterplot consists of a set of points, with each point's position representing the values of the two variables for that data point. For example, a scatterplot might demonstrate the relationship between advertising expenditures and sales revenue or between hours of study and test scores.

A heatmap is a graphical representation that depicts the relationship between two or more variables. It aids in visualizing the relationship between the variables and identifying patterns and trends. A heatmap consists of a set of cells, with the color of each cell representing the value of the variables for that cell. Heatmaps are especially useful for visualizing data organized in a grid or matrix, such as the correlation between various stock prices or the frequency of crime incidents across different times and locations.




##  Correlation

The phrase "correlation does not imply causation" is often used in discussions of statistical relationships, but what exactly does it mean? To answer this question, it is essential to understand the concept of correlation and its role in assessing the connection between two variables.

Correlation is a statistical measure that quantifies the degree of association between two variables. It is often used to describe a linear relationship between the variables. The term "association" is broader than correlation, referring to any relationship between two variables, whether linear or not.

In everyday life, we often observe correlations between events. Various industries, such as finance, healthcare, and marketing, frequently utilize correlation measurements to analyze data and make informed decisions. The word "correlation" can be broken down into "co," meaning together, and "relation," signifying a connection between two quantities.

Correlation can be positive, negative, or uncorrelated. A positive correlation exists when two variables move in the same direction, i.e., an increase in one variable corresponds to an increase in the other. For example, if an increase in advertising expenditure leads to a rise in sales revenue, the variables are positively correlated.

Conversely, a negative correlation occurs when two variables move in opposite directions, with an increase in one variable resulting in a decrease in the other. For instance, a negative correlation may exist between the number of hours spent watching television and academic performance, where increased television time leads to lower grades.

Variables are considered uncorrelated when a change in one variable does not impact the other. Understanding how two variables are correlated enables us to predict future trends and discern the nature of their relationship or lack thereof.

However, it is crucial to remember that correlation does not imply causation. A strong correlation between two variables does not necessarily mean that one causes the other. There may be confounding variables or other factors at play that influence the observed relationship. Therefore, it is essential to consider the context and perform additional analyses before drawing conclusions about causality from correlation alone.


Correlation Analysis: Exploring Relationships Between Variables

Correlation analysis is a statistical method used to study the association or absence of a relationship between two variables. By examining the correlation between variables, researchers can measure the strength and nature of their association.

The primary goal of correlation analysis is to determine a numerical value that indicates the relationship between two variables and how they move together. When a change in one variable is accompanied by a change in another variable, whether direct or indirect, correlation analysis helps quantify the relationship between the two variables.

One of the most popular correlation measures is the Pearson correlation coefficient, which quantifies the strength and direction of a linear relationship between two numerical variables (usually continuous, but not always). When people refer to "the correlation," they usually mean the Pearson correlation coefficient. This coefficient ranges from -1 to 1, with -1 representing a strong negative relationship, 0 indicating no relationship, and 1 signifying a strong positive relationship. For instance, a Pearson correlation of 0.8 between two variables suggests a strong positive relationship, where an increase in one variable typically results in an increase in the other variable.

Beyond the Pearson correlation, other correlation measures include partial correlation, conditional correlation, spatial correlation, and dynamic correlation.

Partial correlation is a statistical method used to measure the relationship between two variables while controlling for the effects of one or more other variables. This technique isolates the relationship between the two variables of interest and examines how it is influenced by the other variables.
For example, when examining the relationship between education level and income, partial correlation could be employed to control for the effects of variables such as age or work experience.

Conditional correlation measures the relationship between two variables while accounting for the effects of one or more categorical variables. Similar to partial correlation, conditional correlation aims to understand the relationship between two variables of interest by considering the influence of other variables.

Spatial correlation assesses the relationship between variables measured at different locations in space. This technique is particularly useful when analyzing data collected across geographical areas, such as climate, population, or economic indicators. Spatial correlation helps determine the extent to which observations close to one another in space are related.

For instance, spatial correlation might be used to study the relationship between pollution levels and population density across various cities.

Dynamic correlation measures the relationship between variables that change over time. This technique is especially valuable when analyzing data collected over extended periods, such as stock prices, economic indicators, or demographic trends. Dynamic correlation estimates the correlation between variables at different points in time and can help researchers understand how the relationship between variables evolves over time.
In summary, correlation analysis is a powerful statistical tool for examining the relationships between variables, allowing researchers to measure the strength and nature of these associations. By employing various correlation measures such as Pearson correlation, partial correlation, conditional correlation, spatial correlation, and dynamic correlation, researchers can gain valuable insights into the complex relationships between variables in numerous fields.

Correlation and Regression

Correlation and regression are two statistical methods used to analyze the relationships between variables. While they share some similarities, their purposes and applications differ significantly.

Similarities:

Both correlation and regression are used to explore relationships between variables.
They both assume a linear relationship between the variables in question.
Both techniques involve the calculation of coefficients that quantify the relationships between variables.

Key Differences:

Correlation measures the strength and direction of the relationship between two variables, whereas regression models and predicts the value of one variable based on the value of another.

Correlation is a descriptive statistical technique, providing a summary of the relationship between variables. In contrast, regression is a predictive statistical technique, employed to make predictions or explain the relationship between variables.

Correlation typically analyzes relationships between two continuous variables, such as height and weight, or age and income. Regression can analyze relationships between two continuous variables but can also assess relationships between a continuous variable and a categorical variable, such as GPA and major.

Correlation is quantified using a correlation coefficient, which ranges from -1 to 1. A coefficient of 0 indicates no relationship, while coefficients of 1 or -1 signify strong positive or negative relationships, respectively. Regression utilizes various statistics, including the coefficient of determination (R-squared), the standard error, and the p-value.

In conclusion, while correlation and regression are related statistical techniques used to explore relationships between variables, they serve different purposes and applications. Correlation is a descriptive technique that measures the strength and direction of a relationship between two variables, while regression is a predictive technique that models and predicts the value of one variable based on another. Understanding these differences is essential when selecting the appropriate statistical method for data analysis.


##  Effect of X on Y / Regression

Economists and social scientists predominantly utilize observational survey data to answer research questions or test hypotheses. By analyzing samples, they make inferences about the larger population. When conducting such analyses, it is crucial to address three key issues:

1. How can factors other than x be allowed to influence y?

2. What is the functional relationship between y and x?

3. How can we ensure that we are capturing a ceteris paribus (all other things being equal) relationship between y and x?

The simple linear regression model addresses these concerns effectively. While this model assumes a linear relationship between x and y, machine learning (ML) techniques do not assume a specific functional form and instead attempt to find the optimal functional form for the best predictive model. In contrast, economists and social scientists are more interested in interpreting the relationship between x and y rather than making predictions.

Victor Chernozhukov's presentation at the 2013 NBER Summer Institute demonstrated that Lasso, a linear method in ML, can better approximate the conditional expectation function (CEF) than ordinary least squares (OLS) for certain datasets.

As Bruce Hansen notes in his recent book, "The OLS regression yields the best linear approximation of the conditional expectation function (CEF). This mathematical property makes regression a favorite tool among economists and social scientists, as it emphasizes the interpretation of an approximation of reality rather than complicated curve fitting."

The predictive power of x on y can be summarized using the conditional expectation function, E(y|xi). This function represents the expected value (population average) of y, given certain covariate x is held constant. It provides a useful representation of how y changes with x. The primary interest lies in the distribution of yi rather than predicting individual yi. The regression CEF theorem states that even if the CEF is nonlinear, regression provides the best linear approximation to it.

OLS estimates the coefficient (β1) using a random sample of data to represent the population and makes assumptions about the relationships between the error term (u) and x. These assumptions include the expected value of the error term being zero in the population, mean independence of the error term, and the zero conditional mean assumption.

In summary, simple linear regression and machine learning methods offer different approaches to understanding the relationship between x and y in observational survey data. While regression emphasizes interpretation and provides the best linear approximation to the CEF, machine learning techniques focus on finding the best functional form for predictive modeling. Both methods have their merits and can provide valuable insights for economists and social scientists.

### How can we estimate the population parameters, $\beta_{0}$ and $\beta_{1}$?

 We find a random sample, which represents the population. We plug observations into the population equation, and use two assumptions $$ E(u)=0 , Cov(x,u)=0$$
 We estimate $$y_{i}=\hat{\beta_{0}}+\Sigma_{1}^{k}\hat{\beta_{k}}x_{i}+u_{i}$$. i.e we minimize sum of squared residuals by solving the following linear regression equation algebraically. This is also known as ordinary least squares(OLS).
$$
		\sum_{i=1}^{n}\hat{u}_{i}^{2}=\underset{\hat{\beta_{0}}, \hat{\beta_{1}}..\hat{\beta_{k}}}{\operatorname{argmin}} \sum_{i=1}^{n}\left(y_{i}-\left(\hat{\beta_{0}}+\hat{\beta_{1}} x_{i}+\hat{\beta_{2}} x_{i}+...+\hat{\beta_{k}} x_{i}\right)\right)^{2}
$$		
		
$$y=X^{'}\beta+u$$, where $$u_{i}\sim(0, \sigma^2)$$ then $$\hat{\beta}= (X^{'}X)^{-1}X^{'}y$$ follows from $$1/n \sum [x_{i}(y_{i} - x_{i}^{'}\hat{\beta})] = 0$$


### Predicting $y$

For any candidates $\hat{\beta}_{0}$ and $\hat{\beta}_{1}$, define a fitted value for each $i$ as		

$$
		\hat{y}_{i}=\hat{\beta}_{0}+\hat{\beta}_{1}x_{i}
$$
		
We have $n$ of these.  $\hat{y}_i$ is the value we predict for $y_{i}$ given that $x=x_{i}$ and $\beta=\hat{\beta}$.
		
The "mistake" from our prediction is called the residual:
$$
		\hat{u}_{i}=y_{i}-\hat{y}_{i}
		=y_{i}-\hat{\beta}_{0}-\hat{\beta}_{1}x_{i}
$$
		
NOTE: Econometric applications typically are not interested in forecasting yi but rather in understanding relationship between $y_{i}$ and some elements of $x_{i}$ with other factors held fixed (ceteris paribus)

We estimate the coefficients by minimizing sum of squared residuals in OLS. Different samples will generate different estimates $(β^1)$ for the true $β_{1}$ which makes $β_{1}$ a random variable. But never forget that β^1 is an estimator of that causal! parameter obtained with a specific sample from the population.  We say the effect of x on y is β1 as long as it is unbiased. We also want our estimates to be consistent, and satisfy asymptotic normality. 

Unbiasedness is the idea that if we could take as many random samples on Y as we want from the population, and compute an estimate each time, the average of these estimates would be equal to β1. OLS is unbiased under the following assumptions.  These assumptions are the population model can be Linear in Parameters, Random Sampling, Sample Variation in the Explanatory Variable, Zero Conditional Mean, Homoskedasticity - Constant Variance (or heteroskedasticity).This tells us that, on average, the estimates will be equal the population values.

Consistency states that if one uses a larger sample size, one reduces the likelihood of obtaining an estimate $\hat{\beta}$ that deviates substantially from the true effect $β$.  Thus, consistency implies that, as a sample size approaches the entire population, an estimate $\hat{\beta}$ is more and more likely to reflect the true estimate $β$. This holds true, for example, when $β$ is unbiased and its variance decreases as the sample size increases.  

Assuming that many samples are drawn randomly, and $β$ is estimated in each sample, asymptotic normality implies that, given the very large size of each sample, the pooled estimate $\hat{\beta}$ obtained from the samples follows a normal distribution. Having this property makes it possible to approximate the distribution of an estimate across many samples well, even if we have only one sample with a sufficient number of observations rather than many samples. In statistical inference, it is very important to understand an estimate's distribution.

Be aware, most of the time, obtaining unbiased $β$ is very hard (even impossible as population DGM is unknown!).  Errors are not iid. For instance, maybe observations between units in a group are related to each other (clustered), non-representative sample, Exclusion or inclusion of variables, Measurement error, Endogeneity, Reverse causality, Missing observations. Thus, most of the time, we can only find associations, or correlations. In this situation, we can only say what is the relationship, association, or correlation between y and x. 


### MLE

Maximum Likelihood Estimation (MLE) is a statistical method used to estimate parameters of a given statistical model based on the observed data. MLE aims to find the parameter values that maximize the likelihood of the observed data under the assumed probability distribution. Unlike Ordinary Least Squares (OLS), which focuses on minimizing the squared differences between the observed and predicted values, MLE seeks to find the parameter values that make the observed data most likely.

In MLE, the main objective is to estimate the unknown parameters of a given statistical model by maximizing the likelihood function. The likelihood function measures the probability of observing the data given the parameters of the model. By maximizing the likelihood function, we find the parameter values that best explain the observed data under the assumed distribution.

MLE has several advantages over other estimation methods, such as OLS, including:

Consistency: As the sample size increases, MLE estimates tend to converge to the true parameter values, assuming that the model is correctly specified. This property ensures that MLE provides reliable estimates as more data becomes available.

Asymptotic Normality: Under certain regularity conditions, the MLE estimates are asymptotically normally distributed. This property allows for the construction of confidence intervals and hypothesis testing using standard statistical techniques.

Efficiency: MLE estimates are asymptotically efficient, meaning they achieve the lowest possible variance among all unbiased estimators. This property ensures that MLE provides the most precise estimates given the available data.

Invariance: MLE estimates are invariant under transformations, which means that if we apply a transformation to the parameter space, the MLE estimate of the transformed parameter will be the same as the transformation of the MLE estimate of the original parameter.

However, MLE also has some limitations, such as:

Dependence on distribution assumptions: MLE requires the assumption of a specific probability distribution for the data. If the assumed distribution is incorrect, the MLE estimates may be biased or inconsistent.

Sensitivity to outliers: MLE estimates can be sensitive to outliers in the data, as they aim to maximize the likelihood of the entire dataset. This sensitivity can lead to biased estimates if the data contains extreme values.

Computational complexity: The maximization of the likelihood function can be computationally intensive, especially for large datasets and complex models.

In summary, Maximum Likelihood Estimation (MLE) is a powerful statistical method for estimating the parameters of a given statistical model based on the observed data. It provides a flexible and efficient approach to parameter estimation under the assumption of a specific probability distribution. However, it is essential to consider the limitations of MLE when applying it to real-world data and carefully assess the distributional assumptions and the presence of outliers in the data.

##  Causal Effect

You have most likely heard the term “Correlation is not
causation”, which means, loosely that “just because two things
happen together, doesn't mean that one of them caused the
other.” A better term is “correlation is not sufficient for causation.”There are some things we can do to make causal inference
possible, but they happen before the sample is taken.
This is a big change from most of the statistics you learn.
Usually, you're given numbers from some random sample, and
you don't have any control over that sample. You just have to
take the numbers and make sense of them.
In the design phase, you decide how treatments are going to
be assigned to sample units/patients. If you can't directly
assign treatments, you need to collect data about covariates.

"You may have heard the phrase "correlation does not equal causation," which means that just because two things are related or happen together does not necessarily mean that one of them caused the other. A more accurate statement would be "correlation is not enough to establish causation." To make causal inferences, there are certain steps that must be taken before collecting data, such as designing how treatments will be assigned to sample units or patients. In contrast, statistical analysis typically involves working with data that has already been collected, and the focus is on understanding and interpreting the relationships between variables within the given data set. It is important to keep in mind that correlation does not necessarily imply causation, and other factors may be at play when trying to understand the relationships between variables."

Most of the time though, we also want to find out not only the relationship or correlation between observations but also the reason behind it. Using this information, we are able to take action to alter the relationships and causes. Essentially, we want to know what the consequences are of not doing one thing or the other. Our goal is to understand the effects and consequences of specific actions, policies, and decisions.
In order to develop a hypothesis about such causal effects, we rely upon previous observations, personal experiences, and other information. Researchers in economics, health, and social sciences analyze empirical data to evaluate policies, actions, and choices based on quantitative methods.

When we want to say something about why? We work with potential outcomes Potential Outcomes (Neyman, 1923 - Rubin, 1975) framework, especially in applied economics, or directed acyclic graphs (DAG) which you can think presenting a chain of causal effects with a graph. We will review causal analysis as short as possible, and considering the approaches integrating the machine learning with causal analysis. The books we can recommend in this topic are Causal Inference-The Mixtape, The Effect, Mostly Harmless Econometrics,...

At the center of interest is the causal effect of some intervention or treatment on an outcome of interest by inferring what the outcome would be in the presence and in the absence of a specific treatment. Causality is tied to a unit (person, firm, classroom, city) exposed to an action (manipulation, treatment or intervention) at a particular point in time. The Causal Effect is the comparison or difference of potential outcomes,$Y_{i}(1)$ and $Y_{i}(0)$, for the same unit, at the same moment in time post-treatment. where $Y_{i}(1)$ is the outcome when unit $i$ exposed the treatment (active treatment state), and $Y_{i}(0)$ is the outcome when same unit $i$ has not exposed the treatment (control state) (at the same point in time)

Let's say $D$ is the treatment indicator (or intervention, or policy). When $D=1$ , the unit receives the treatment or participates in the intervention, thus these units constitute "treatment group". When $D=0$ , the unit does not receive treatment or does not participate in the intervention, thus these units constitute "control group". 

The causal effect of treatment,$D$, on outcome,$Y$, is the difference between the potential outcomes $Y_{i}(1)- Y_{i}(0)$ for unit $i$. 

However, we can not observe the unit in 2 different state at the same time. We can not observe "The Road Not Taken" (by Robert Frost). "The fundamental problem of causal inference" is therefore the problem that at most one of the potential
outcomes can be realized and thus observed (Holland, 1986). Thus, Holland (1986, 2003) says "No causation without manipulation" 

 Keep in mind that counter-factual state is and never will be observable. We can define counterfactual as what would have happened in the absence of a policy/treatment. Donald Rubin has been known to say that "causal inference is a missing data problem" (Ding and Li, 2018)
Hence, there are several methods used to find causal effect. Experiments (Randomized Control Trials) and Quasi-natural Experiments such as Regression Discontinuity (Sharp, Fuzzy), Instrumental Variable, Difference-in-Difference(-in-Difference), Synthetic Cohort, Propensity Score, and Partial identification. 

We can never know the real! causal effect (of a unit) in social sciences. Using Rubin’s framework, we can only estimate the causal effect under certain assumptions. Overcoming the missing data problem arising from the fact that only one state of nature is realized is very difficult. To do so requires credible assumptions!


Main implicit assumption in Rubin framework for all the aforementioned methods is the Stable Unit Treatment Value Assumption (SUTVA, Rubin 1978). SUTVA implies that potential outcomes of observation $i$ are independent of the treatment assignment of all other units. In another word, the unit's potential outcome are not affected by the spillover or interference effects by the treatment of other units. Thus, SUTVA rules out general equilibrium or indirect effects via spillovers.  Moreover, SUTVA also implies that the treatment, $D$, is identical for all observations and no variation in treatment intensity. Most of the current empirical work assumed this assumption satisfied. However, it may not be always plausible to assume no spillover effect when treated group is very large share of the population. We also need to reassess our policy proposal, which are based on the findings of randomized or natural experiments, for all population as they may violate SUTVA assumption when these studies are designed with small sample. 



### Average Treatment Effect(ATE)


While causal effects cannot be observed and measured at a unit level, under certain statistical assumptions we may identify causal effect of treatment,D, on outcome,Y, at an aggregate level, while using treatment and control groups.  

Even though we cannot observe treatment effect for each individual/unit separately, we know that for a given population these individual causal effects-if exist- will generate a distribution of treatment effects. [footnote: A distribution is simply a collection of data, or scores, on a variable. Usually, these scores are arranged in order from smallest to largest and then they can be presented graphically. — Page 6, Statistics in Plain English, Third Edition, 2010.]Thus, we can estimate the mean, variance and other parameters related to this distribution. 

 Most common parameter we want to estimate is the mean of treatment distribution. We can think the average treatment effect(ATE) is the population average of all individual treatment effects. We can formally write:
$$	
		 \delta^{ATE} = E[\delta_{i}]= E[Y_{i}(1) - Y_{i}(0)] = E[Y_{i}(1)] - E[Y_{i}(0)]
$$

 As known, the expected value of a random variable X is often denoted by E(X). 
 
 
 Assume government implement a policy, an agency act, or a doctor prescribe a pill. All in all, we can think 2 states. Treatment state in which every units in the population exposed a treatment, and control state in which every units in the population has not exposed the treatment . The equation above shows that the average of the outcome for everyone in treatment state and the average of the outcome for everyone in control state is called the average treatment effect for all population.
 
Depends on the question, we may want to estimate different treatment effects as well. Some of them are:
  
 Average Treatment Effect on the Treated (ATT, or ATET): The average treatment effect on the treatment group is equal to the average treatment effect conditional on being a treatment group member (D=1).
 
$$ 
 ATT= E[Y_{i}(1)|D=1] - E[Y_{i}(0) | D=1]
 $$
Average Treatment Effect on the Untreated (ATU): The average treatment effect on the control group is equal to the average treatment effect conditional on being untreated.
$$ 
 ATU= E[Y_{i}(1)|D=0] - E[Y_{i}(0) | D=0]
 $$

However,we want to emphasize ATE (ATT, and ATU) is unknowable because of the fundamental problem of causal inference.i.e. we can only observe individuals either when they receive treatment state or when they do not, thus we cannot calculate the average treatment effect. However, we can estimate it. How?

If we can find a population or split the population such as some of whom receive this treatment or act on it and some of whom has not. Then we can estimate "causal/treatment effect" as the difference between average outcome of the treated group and average outcome of control group.

How does the population split into treatment and control group? Whether units have any choice to be eligible for the treatment or not? Whether splitting process has any direct effect on outcome or not? How large, how many and how similar similar these groups? Whether treatment level is equal or not? Whether everyone who are eligible for treatment receive treatment or not? 
Answers of all these questions require different identification strategies and leads all the different causal methods we use.
 
### Additional Treatment Effects

yesil pdf fileindan
Local Average Treatment Effect (LATE): g = E[dijCompliers]The Local Average Treatment Effect (LATE) is a statistical measure that is used to estimate the effect of a treatment or intervention on a specific subgroup of the population. It is a measure of the average treatment effect for a group of individuals who would not have received the treatment if they had not been eligible for it.

LATE is typically estimated using a technique called instrumental variables (IV) regression, which involves using a variable that is correlated with the treatment but is not directly affected by the outcome of interest. By using this "instrumental" variable, researchers can estimate the effect of the treatment on the subgroup of individuals who are eligible for the treatment based on their values of the instrumental variable.

LATE is a useful measure when the treatment effect is not the same for all individuals in the population. For example, a study might estimate the LATE of a new medication on individuals with a specific type of disease, in order to understand how the medication affects this subgroup compared to the overall population.

It is important to note that LATE is only applicable when the treatment is randomly assigned, meaning that individuals are assigned to receive the treatment or not receive the treatment based on chance rather than their characteristics or other factors. This is necessary in order to ensure that the treatment effect can be accurately estimated and is not confounded by other factors.



Conditional Average Treatment Effect (CATE)
d(x) = E[Yi(1)−Yi(0)jXi = x] = E[dijXi = x]
• Xi exogenous pre-treatment covariates/features
• Xi includes not only confounders but also other covariates which are potentially
responsible for effect heterogeneity
• CATEs are often called individualised or personalised treatment effects
• CATEs can differ from CATET, r(x), and CLATE, g(x)

The Conditional Average Treatment Effect (CATE) is a statistical measure that is used to estimate the effect of a treatment or intervention on a specific subgroup of the population. It is a measure of the average treatment effect for a group of individuals who are similar in some way, such as having the same level of a certain characteristic or risk factor.

CATE is typically estimated using a technique called conditional mean regression, which involves estimating the expected value of the outcome variable for individuals with different values of the conditioning variable. By using this conditioning variable, researchers can estimate the effect of the treatment on the subgroup of individuals who are similar in terms of the conditioning variable, compared to the overall population.

CATE is a useful measure when the treatment effect is not the same for all individuals in the population and when there is a characteristic or factor that may influence the treatment effect. For example, a study might estimate the CATE of a new medication on individuals with a specific type of disease, in order to understand how the medication affects this subgroup compared to individuals with a different type of disease.

It is important to note that CATE is only applicable when the treatment is randomly assigned, meaning that individuals are assigned to receive the treatment or not receive the treatment based on chance rather than their characteristics or other factors. This is necessary in order to ensure that the treatment effect can be accurately estimated and is not confounded by other factors.

Group Average Treatment Effects (GATEs):
d(g) = E[d(x)jGi = g]
where the groups g can be defined based on exogenous or endogenous
variables

Group Average Treatment Effects (GATE) is a statistical measure that is used to estimate the effect of a treatment or intervention on a specific subgroup of the population. It is a measure of the average treatment effect for a group of individuals who are similar in some way, such as having the same level of a certain characteristic or risk factor.

GATE is typically estimated using a technique called group mean regression, which involves estimating the mean of the outcome variable for individuals with different values of the grouping variable. By using this grouping variable, researchers can estimate the effect of the treatment on the subgroup of individuals who are similar in terms of the grouping variable, compared to the overall population.

GATE is a useful measure when the treatment effect is not the same for all individuals in the population and when there is a characteristic or factor that may influence the treatment effect. For example, a study might estimate the GATE of a new medication on individuals with a specific type of disease, in order to understand how the medication affects this subgroup compared to individuals with a different type of disease.

It is important to note that GATE is only applicable when the treatment is randomly assigned, meaning that individuals are assigned to receive the treatment or not receive the treatment based on chance rather than their characteristics or other factors. This is necessary in order to ensure that the treatment effect can be accurately estimated and is not confounded by other factors.
 
### Selection Bias and Heteregeneous Treatment Effect Bias:

 When a group of individuals receive a treatment and a group does not, most inclined to calculate the treatment effect just calculating the simple difference between average outcomes of treated group and control group. However, this is (nearly) always wrong.
 
Can we find average treatment effect by calculating the simple difference between the average outcome for the treatment group and the average outcome for the control group? 

There may be already intrinsic differences between these 2 groups as some already decided to choose treatment, or there may be differences with some other characteristics that will already effect outcome directly or through another path. Hence, all of these will be included in the simple difference between average of outcome of these 2 groups. That "misassigned" effect is called as treatment selection bias. We can think the selection bias as the difference between a treatment group and a control group if there was no treatment at all. We want to emphasize that we may not observe this difference, we may not verify that. However, the main purpose of all causal inference methods is to eliminate as much as possible this bias by imposing different identifying assumptions.

Heteregeneous treatment effect bias always exist if we want to calculate ATE. However, when we assume that treatment effects -dosage effect- are constant then this bias disappears. Even though this is a strong assumption, this is very common and plausible in social sciences and economics as we want to analyze average effects, not individual effect. That average treatment/causal effect is presented either treatment effect for average person or "homogeneous" average treatment effect for everyone. However, heterogeneous treatment effect and dealing its bias is one of the major topic in which machine learning methods are contributing recently.  

Decomposition of difference in means
$$
		\begin{eqnarray*}
			\underbrace{E[Y_{i}(1) | D=1] - E[Y_{i}(0) | D=0]}_{{\text{Simple Difference in Outcomes}}}&=& \underbrace{E[Y_{i}(1)] - E[Y_{i}(0)]}_{{\text{Average Treatment Effect}}} \\
			&&+ \underbrace{E[Y_{i}(0)|D=1] - E[Y_{i}(0) | D=0]}_{{\text{Selection bias}}}  \\
			&& + \underbrace{(1-\pi)(ATT - ATU)}_{{\text{Heterogenous treatment effect bias}}} 
		\end{eqnarray*}
		$$
		where $(1-\pi)$ is the share of the population in the control group.(Detailed derivation of this equation in Mixtape page 131-133)

As we mentioned the simple difference between the average outcome for the treatment group and the average outcome for the control group can be assumed by most as an average treatment effect. It may be true only if we do not have selection and heterogeneous treatment bias. However, most of the time already the difference exist between a treatment group and a control group before treatment implemented. Thus selection bias exists. 

Most of the time the treatment effects individuals as well as groups differentially. Thus, the average effect of treatment for the group consist from treated individuals and for the group consist from untreated individuals differ. The multiplication of that difference and the share of the population in the control group is called as Heterogenous treatment effect bias.

As previously noted, we are unable to directly observe individuals in both treatment and control states, making it impossible to explicitly calculate treatment effects and associated biases. Social scientists have been devising strategies to address these biases and estimate treatment effects, with machine learning methods contributing to these advancements in recent years. The various methodologies can be categorized as follows:

Regression, penalized regression, and fixed effects

Matching and propensity score methods

Randomization inference

Instrumental variables, difference-in-differences, regression discontinuity, and event studies

Synthetic control method

Causal forest method

In the upcoming chapters, we will delve into these approaches and explore the relevant machine learning techniques that complement and enhance these methods in estimating treatment effects.

