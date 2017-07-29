# Enron Email Dataset Investigation


This project is 6th of the data analyst nano degree of Udacity, Intro to Machine Learning. 


The goal is to build a machine learning capable of catch the person on interest in the Enron database. As person of interest we meant the people that were , or supposed, involved in the Enron bankruptcy and caused it. 

The dataset is quite small and contain only few poi , and 146 line. Features are monetary, as salary and other sources of income, but as well information about the mail, like email sent to poi or received ans so on. 


##Project Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.


### Features 

####Financial features 

- bonus 
- deferral payments  
- deferred income 
- director fees
- exercised stock options 
- expenses
- loan advances
- long term incentive
- other
- restricted stock
- restricted stock_deferred
- salary
- total payments
- total stock value

(all units are in US dollars)

####Email features

- email address
- from messages 
- from poi to this person 
- from this person to poi
- to messages 
- shared receipt with poi

(units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label

- poi 

(boolean, represented as integer)



## References 


- http://scikit-learn.org/stable/index.html
