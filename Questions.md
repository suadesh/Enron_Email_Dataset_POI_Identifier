# Enron Email Dataset Investigation

## Student : Massimiliano Z D'Adamo
## Udacity NanoDegree Data Analyst 



## 1 ENRONN DATASET 

This Dataset is composed of different informations of people working at __Enron__ before the bankruptcy. We have multiple data about the salary and other kind of income of the most importants figures of Enron. Also, from the entire email database, we have other informations about the amount of email sent and received for, and to, each person. The dataset is composed of __146__ data point, but not all of them were convicted, or supposed involved in the facts that ruined Enron in less than a month. There are only __18__ person of interest and 128 Not Person of interest present in this dataset,  This represent the __12.33%__ of POI in the dataset.



![](https://github.com/suadesh/Enron_Email_Dataset_POI_Identifier/blob/master/POI.png?raw=true)


There are also __1358__ missing values all over the dataset, that will be transformed into 0.
In details the count of each missing features, as we can see 20 out of 21 features have a missing values, the only features that have no missing values is POI.  

Financial Feature | missing |   | Email Features | Missing |  | POI LABEL | Missing
|---|---|---|----|---|---|---| ---|
bonus | 63 | | email address |33 | | POI | 0 
deferral payments  | 106 | |from messages | 58
deferred income | 96 | |from poi to this person |  58
director fees | 12 | | from this person to poi |  58
exercised stock options  | 43 | | to messages | 58
expenses |50 | | shared receipt with poi | 58
loan advances | 141
long term incentive | 79
other | 53
restricted stock | 35
restricted stock_deferred | 127
salary |50
total payments | 21
total stock value  | 19
 
The goal is to build a POI identifier, with machine learning ,capable of individuate this person of interest (poi), for the data available.  In order to achieve this , and have a great result, we need to analyse correctly this dataset, choose the proper features , and pick and tune the right algorithm. 
First of all this dataset contains one major outliers, and it was the line *TOTAL*, the sum of all other line, that I just simply remove. An other datapoint that i will remove because, again, is not a person is *THE TRAVEL AGENCY IN THE PARK*. 
 
There are other outliers for some data, especially for stock values features, but in order to do not lose to many informations offered form this dataset of only 144 datapoint , I will keep these outliers.   


----------

## 2 Features Creation and Selection 


I decided to create 2 new features, that are the percent of email received from poi and sent to poi , over all the emails sent and received. I thought of these features, because I wanted to create a feature that represented better the quantity of emails send and received over all the emails.              
The first one is __percent to poi__ and is *from this person to poi* divided by the sum of *from this person to poi* and *from messages*.               
The second features is __percent from poi__ and is *from poi to this person* divided by the sum of *from poi to this person* and *to messages*. 
Here the initial scores with select K best of all the available features.

### Select K best score initial available features 

Financial Feature | missing |   | Email Features | Missing |
|---|---|---|----|---|---|---| ---|
bonus | 21.0600017075 |   |from messages | 0.164164498234
deferral payments  | 0.21705893034 | |from poi to this person |  5.34494152315
deferred income | 11.5955476597 | | from this person to poi |  2.42650812724
director fees | 2.10765594328 | | to messages | 1.69882434858
exercised stock options  | 25.0975415287 | | shared receipt with poi | 8.74648553213
expenses | 6.23420114051 |  | percent to poi | 0.015052574699
loan advances | 7.24273039654 || percent from poi | 5.43963902436
long term incentive | 10.0724545294 | 
other | 4.24615354068
restricted stock | 9.34670079105
restricted stock_deferred | 0.0649843117237
salary | 18.575703268
total payments | 8.87383525552
total stock value  | 24.4676540475


First of all I did not select the features 'email address', logically is only the email address of each person. 

And after have plotted the features , I decided to start selecting only features that did not contains to many missing values, so I drop all the features with more than 60 missing values. This is the first list : 

- poi
- exercised stock options
- expenses
- from messages
- from poi to this person
- from this person to poi
- other
- total stock value
- percent from poi
- percent to poi
- restricted stock
- shared receipt with poi
- to messages
- total payments
- salary

After this I decided to use RFECV , combined with AdaBoost Algorithm , and StratifiedShuffleSplit in order to obtain the best number of features.

![](https://github.com/suadesh/Enron_Email_Dataset_investing/blob/master/Features.png?raw=true)

The final choice so were 7 features plus poi ( on the right the features importance with the algorithm chosen 

- poi
- exercised stock options - 0.14
- expenses - 0.23
- from message - 0.06
- from this person to poi - 0.15
- other - 0.29
- shared receipt with poi - 0.13


### Impact on the final scores with and without the new features created 

Features | Accuracy | Precision | Recall | F1 
---|---|---|---|---
Final Selection without the new created features  | 0.87550 | 0.59081 | 0.41800 | 0.48960 
Final Selection with percent from poi | 0.87313 | 0.53492 | 0.37150 | 0.43848
Final Selection with percent to poi| 0.88120 | 0.57476 | 0.41900 | 0.48467
Final Selection with both the new created features| 0.87140 | 0.52464 | 0.37800 | 0.43941 

As we can see the accuracy reach a better accuracy and recall score using also *percent to poi*, but it cannot be said for precision. 


--------


## 3 Final choice of algorithm 

I chose to use AdaBoost, that was the one that gave me the better result. I tried also Decision Tree, Naive Bayes and  RandomForest. For Naive Bayes and Random forest I had good score for precision but on the other hand a low recall score. Decision tree was both more close scores in precision and recall, but lower from my final choice.  

Here a small resume of the best one that I tried:
 
Algorithm | Accuracy | Precision | Recall | F1 
---|---|---|---|---
Ada Boost | 0.87550 | 0.59081 | 0.41800 | 0.48960 
Naive Bayes | 0.83829 | 0.36720 | 0.18250 | 0.24382
Decision Tree | 0.82207 | 0.37180 | 0.35600 | 0.36373
Random Forest | 0.85093 | 0.43106 | 0.13600 | 0.20677 


---------

## 4  Tuning the parameters of the algorithm 

The most of the algorithm used in machine learning have specific parameters , that can help to adapt to the specific case. If the parameters are not well chosen,  there are two risks: to do not fit very well the data , or to fit too much. In the first case the result will be not very accurate, compromising the result. In the second case the risk is to fit so well the dataset and draw a decision boundary too specific to the data, useless with other dataset. 
To start I used the default parameters, than I modify some parameters.  

I finally used GridSearchCV to look for the best one through a matrix of choices.  At the end I modified only the learning rate at 0.9 and the numbers of estimators at 100. 



----------

## 5 Validation


Validate our algorithm we need to evaluate the metrics of the performance, like accuracy , recall and precision score.        
This will tel us how our algorithm is working and how it is capable of labelling, poi in this case. 
The classic way of working is to split the dataset in two , one part to train the algorithm and the second to perform a test and obtain the scores. 
When dividing clearly we lost some informations for the training because they now belong to the test set, and vice-versa. In order to obtain a more consistent result, we use often use cross validation methods. For example, __K-fold__ cross validation, divide the data set in k group, pick on group for test, train on all the others and finally perform testing on the test group obtaining the scores metrics. It then performs this procedure for all the k group, obtaining k scores. The final result than is the average of all these scores.
Clearly this will increase test and training time, but on the other hands it will increase the accuracy 

In this study I used __Stratified Shuffle Split__ , that is a merge of Stratified K Fold and Shuffle Split, which returns stratified randomised folds , in order to preserve the percentage of samples for each class. This cross validation was used to found the best number of features and in Grid Search for the best algorithm parameters. 

To validate my choices, i import the test_classifier for the tester.py file, that uses Stratified Shuffle Split. 

------------

## 6 Metrics and performance


For this project I ended up using AdaBoost Algorithm , and this is the result that I had: 

Algorithm | Accuracy | Precision Score | Recall Score 
---|---|---|---
Ada Boost | 0.87550 | 0.59081 | 0.41800 
        

__Precision Score__ represents the capacity of the algorithm to do not label positive when is actually negative. 
The higher is this score, and more confident we are that the datapoint flagged as positive is really positive.

__Recall score__, is the  ability of the classifier to find all the positive samples. The higher is the score and the most probable is that among all the actuals positives datapoint, the algorithm flagged the most correctly. 

In this study, a higher precision score would means that when the algorithm flags a datapoint as POI, I'm quite confident that it is a POI. On the other hand an higher recall score would menas that the algorithm was capable  to labels POI the most of all the actual POIs in the data set. 

- If I wanted to be really sure before labelling someone a POI, I would like to have an higher precision score, on the contrary with a lower precision score, I would not be so confident when flagging someone.

- If I wanted to flag the most of the POIs, I would like an higher Recall Score, on the other hand a lower score would means that the algorithm did not catch many POIs. 


__Accuracy__ is the sum of __true positive__ and __true negative__ (the total of the corrected predictions ) divided by the total of the predictions. 


 

