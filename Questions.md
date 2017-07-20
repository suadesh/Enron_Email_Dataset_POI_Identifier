# Questions 




####1 
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

This Dataset is composed of different informations of people working at Enron before the bankruptcy. We have multiple data about the salary and other kind of income of the most importants figures of Enron. Also, from the entire email database, we have other informations about the amount of email sent and received for, and to, each person. The dataset is composed of 146 persons, but not all of them were convicted, or supposed involved , in the facts that ruined Enron in less than a month, only 9 are present in this dataset.   
The goal is to build a POI identifier, with machine learning ,capable of individuate this person of interest (poi), into other similar dataset. In order to achieve this , and have a great result, we need to analyse correctly this dataset, chose the proper features , and pick and tune the right algorithm. 
First of all this dataset contains one major outliers, and it was the line Total, the sum of all other line, that I just simply remove. 
There other outliers for some data, especially for stock values features, but in order to do not lose to much information in a dataset of only 146 line, i will perform a scaling min and max of these features.  


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
I also try SelectKbest, in order to obtain a result


####3
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]



####4 
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

####5 
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

####6
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]