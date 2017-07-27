# Questions



####1 
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

This Dataset is composed of different informations of people working at __Enron__ before the bankruptcy. We have multiple data about the salary and other kind of income of the most importants figures of Enron. Also, from the entire email database, we have other informations about the amount of email sent and received for, and to, each person. The dataset is composed of __146__ data point, but not all of them were convicted, or supposed involved in the facts that ruined Enron in less than a month, only __18__ are present in this dataset. This represent the __12.33%__ of the dataset. There are also __1358__ missing values all over the dataset, that will be transformed into 0.
 
The goal is to build a POI identifier, with machine learning ,capable of individuate this person of interest (poi), for the data available.  In order to achieve this , and have a great result, we need to analyse correctly this dataset, choose the proper features , and pick and tune the right algorithm. 
First of all this dataset contains one major outliers, and it was the line *TOTAL*, the sum of all other line, that I just simply remove. An other datapoint that i will remove because again is not a person is *THE TRAVEL AGENCY IN THE PARK*. 
This change a little bit the dataset that now contains __144__ datapoint, __18__ POIs, that represent __12.5%__, and __1334__ missing points.   

There are other outliers for some data, especially for stock values features, but in order to do not lose to many informations offered form this dataset of only 144 datapoint , I will keep this outliers.   


----------

####2 
What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]


For this project I used theses features :

+ exercised stock_options
+ total stock value
+ expenses
+ deferred income
+ salary
+ from this person to poi
+ from poi to this person
+ percent to poi
+ percent from poi

I use different way to select them. First of all, I look at it in the spreadsheet, secondly I plot them , and them I chose some of them. I eventually modify the selection later when I chose which algorithm use. 
I tried SelectKbest, in order to obtain a better result, but in the end I kept my choice. I also tried scaling only some features, using minmaxscaler, but I did not find benefit of using it. 

I decided to create 2 new features, that are the percent of email received from poi and sent to poi , over all the emails sent and received. I thought of these features, because I wanted to create a feature that represented better the quantity of emails send and received over all the emails. 


VARAIBLE NAME | IMPORTANCE | Equation 
---|---|---
PERCENT TO POI| __0.7__ | (from this person to poi)/[(from this person to poi)+(from message)]
PERCENT FROM POI | __0.13__ |  (from poi to this person)/[(from poi to this person)+(to message)]



------------

####3
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I chose to use AdaBoost, that was the one that gave me the better result. I tried also Deciosion Tree, Naive Bayes and  RandomForest. I had a good result with naive Bayes, and low training time but result with AdaBoost were better in the end, and made me chose this one. 

Here a small resume of the best one that I tried:
 
Algorithm | Accuracy | Precision | Recall | F1 
---|---|---|---|---
Ada Boost | 0.88467 | 0.59221 | 0.43350 | 0.50 
Naive Bayes | 0.87373 | 0.53576 | 0.39700 | 0.45606
Decision Tree | 0.83493 | 0.38992 | 0.42150 | 0.40509
Random Forest | 0.86373 | 0.46901 | 0.16650 | 0.24576 


---------

####4 
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]


The most of the algorithm used in machine learning have specific parameters , that can help to adapt to the specific case. If the parameters are not well choses there are to risk, to do not fit the , with the algorithm the data , and on the other hand , to fit too much and overfitting. In the first case the result will be not very accurate, compromising the result. In the second case the risk is to fit so well the dataset, to draw a decision boundary too specific to data, that actually a simple one would gave a better result. 
To start I used the default parameters, than I modify some parameters.  I used GridSearchCV to look for the best one through a matrix of choices.  At the end I modified only the learning rate setting it at 0.9 and the numbers of estimators at 100. 



----------

####5 
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

To validate data we always split the dataset in two, in order to obtain a training set and a test set. First of all this will allow us to train our machine on the training set and than receive estimation of our performance on the test set, an independent dataset. Secondly it serves as verifying of overfitting.            
After using cross validation that split our data set in two, or actually in 4, Features training and test, labels training and test. I train my algorithm with the trainings dataset, using the attribute __.fit__ of the classifier and than making prediction with the attribute __.prediction__.      
I than compare the prediction obtained with the actual values, labels test, using the attribute __.score__ to estimate the accuracy, but also the functions __recall score__ and __prediction score__. 



------------

####6
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]


For this project I ended up using AdaBoost Algorithm , and this is the result that I had: 

Algorithm | Accuracy | Precision Score | Recall Score 
---|---|---|---
Ada Boost | __0.88467__ | __0.59221__ | __0.43350__ 
        

__Precision Score__ represents the capacity of the algorithm to do not label positive when is actually negative. 
The higher is this score, and more confident we are that the datapoint flagged as positive is really positive, and that the algorithm commits few error when flags someone. 

__Recall score__, is the the ability of the classifier to find all the positive samples. The higher is the score and the most probable is that among all the actuals positives datapoint, the algorithm flagged the most correctly. 

In this study, an higher precision score would means that when the algorithm flags a datapoint as POI, I'm quite confident that it is a POI. On the other hand an higher recall score would menas that the algorithm was capable  to labels POI the most of all the actual POIs. 

- If I wanted to be really sure before labelling someone a POI, I would like to have an higher precision score, on the contrary with a lower precision score, I would not be so confident when flagging someone.

- If I wanted to flag the most of the POIs, I would like an higher Recall Score, on the other hand a lower score would means that the algorithm did not catch many POIs. 


__Accuracy__ is the sum of __true positive__ and __true negative__ , divided by the sum of all four of them(the total of predictions ). So Accuracy it the percent of all corrected predictions divided by all predictions. 



