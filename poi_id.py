#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np

sys.path.append("../tools/")
#sys.path.append(
#        "/Users/massi/Google_Drive/Perso/Data_Science/Udacity/ud120-projects-master/tools/")
#sys.path.append(
#        "/Users/massi/Google_Drive/Perso/Data_Science/Udacity/ud120-projects-master/Enron_Email_Dataset_investing/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB 
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'expenses',
                 'deferred_income',
                 'salary',
                 'from_this_person_to_poi',
                 'from_poi_to_this_person',
                 'percent_to_poi',
                 'percent_from_poi']
                 
                    # You will need to use more features



### Load the dictionary containing the dataset
with open(
        "/Users/massi/Google_Drive/Perso/Data_Science/Udacity/ud120-projects-master/Enron_Email_Dataset_investing/final_project_dataset.pkl", 
        "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print "the data set is composed of ", len(data_dict), " elemets and I wil use 9 features" 

'''
In this way I will remove Total that is not a person but the total in the
dataset of all the employees. 
'''
#print data_dict['TOTAL']
data_dict.pop('TOTAL')


''' Possibile outliers that are keep in the data set. 
their value is imporant in training, most of them are poi. I tried without 
using them, once at the time and the result are worst than with them. 
data_dict.pop('LAY KENNETH L')
data_dict.pop('HIRKO JOSEPH')
data_dict.pop('SKILLING JEFFREY K')
data_dict.pop('PAI LOU L')

'''


''' Features Available to investigate 
'salary','to_messages','deferral_payments','total_payments','deferred_income',
'exercised_stock_options','bonus','restricted_stock','expenses',
'shared_receipt_with_poi', 'restricted_stock_deferred','total_stock_value',
'loan_advances','from_messages','other','from_this_person_to_poi',
'director_fees','long_term_incentive','from_poi_to_this_person'

'''

'''
This procedure will create 2 new variables, that are in a sort of way the 
percentage of the email received and sent to poi over the total emails. 
I will call them "percent_to/from_poi" 
'''
for key, values in data_dict.items():
    lista = [values['from_this_person_to_poi'],values['to_messages'],
             values['from_poi_to_this_person'],values['from_messages']]
    valori = []
    for k in lista:
        if k == 'NaN':
            k = 0
            valori.append(k)
        else:
            valori.append(float(k))
    if valori[0] == 0 and valori[1] == 0: 
        pass
    else:
        percent_to_poi = valori[0]/(valori[1]+valori[0])
    if valori[2] == 0 and valori[3] == 0 : 
        pass
    else:
        percent_from_poi = valori[2]/(valori[3]+valori[2])
    values['percent_to_poi']= percent_to_poi
    values['percent_from_poi']= percent_from_poi


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

'''
With this procedure I tested the scaling of only fews features, but the
 result obtained was worst than before 
 ( kept as remainder for further investigation )

b = data[:,4]
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(b)

f = 4
#g = 0      
#for f in range(1,3):
j = 0 
for j in range(0,144):
    data[j][f] = X_train_minmax[j]
    j+=1
 #f +=1 
  #  g +=1 

'''
labels, features = targetFeatureSplit(data)

      
'''

This is the last procedure used to plot with matplotlib the data" 
i = 1 

for i in range(9):
    for point in data:
        AA = point[0]
        BB = point[i]
        matplotlib.pyplot.scatter( AA, BB )

    matplotlib.pyplot.xlabel(features_list[0])
    matplotlib.pyplot.ylabel(features_list[i])
    matplotlib.pyplot.show()
    i +=1 

for point in data:
    salary = point[1]
    exercised_stock_options = point[2]
    matplotlib.pyplot.scatter( salary, exercised_stock_options )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("exercised_stock_options")
cd matplotlib.pyplot.show()

''' 

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

clf= AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                        learning_rate=0.9, n_estimators=100, random_state=None)



'''
much faster than Adaboost but result are slightly worst
clf = GaussianNB() 


USED TO FIND THE CORRECT PARAMENTERS FOR ADABOOST 
parameters = {'learning_rate':[0.5,0.7,0.8,0.9,1.0,1.1,1.2], 
                 'n_estimators':[50,75,100,150,200]}
clf = GridSearchCV(adr, parameters)
clf=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                      learning_rate=1.0, n_estimators=75, random_state=None)

OTHER CLASSIFIER TESTED, RESULT WERE NOT AS GOOD, PROBABLY TO BE TUNED BETTER 
BUT IN THE END THE CHOISE HAS GONE TO ADABOOST 

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion ='gini',splitter='best')



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)

'''



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-lcearn.org/stable/modules/generated/sklearn.cross_validation.
### StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

'''

Tryed pipelie with MinMaxScaler and selectKbest but result do not improve. 
scaler = preprocessing.MinMaxScaler()
skb = SelectKBest(k=8).fit_transform(features_train, labels_train)
clf =  Pipeline(steps=[("SKB", skb),("Adaboost", ada)])


min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(features_train)

'''

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,features_test , labels_test)
print 'Score Mean: ', scores.mean() 

print 'Accurancy:', clf.score(features_test,labels_test)

print 'Precision Score :' , precision_score(labels_test,pred)
print 'Recall Score :', recall_score(labels_test,pred)


'''
using the attribute to see the feature importance. 


importance =  clf.feature_importances_

i = 1 
for p in importance:
    print features_list[i] , "importance : " ,  p 
    i +=1 

'''
#clf.best_estimator_

#rsult = clf.cv_results_
#print "features importance :" , clf.feature_importances_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)