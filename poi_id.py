#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np

sys.path.append("../tools/")
sys.path.append("/Users/massi/Google_Drive/Perso/Data_Science/Udacity/ud120-projects-master/tools/")

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


################################################################################################


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
'''
HAND PICKED FEATURES 

features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'expenses',
                 'deferred_income',
                 'salary',
                 'from_this_person_to_poi',
                 #'from_poi_to_this_person',
                 #'percent_to_poi',
                 'percent_from_poi']
                 
'''                 


### COMPLETE FEATURES LIST 
features_list = ['poi',
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'deferred_income',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'expenses',
                 'shared_receipt_with_poi',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'loan_advances',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                 'long_term_incentive',
                 'from_poi_to_this_person',
                 'percent_to_poi',
                 'percent_from_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

## INVESTIGATION OF THE DATA SET AND THE MISSING VALUES 
# In the following lines I  will look deeper into the dataset 
npoi = 0.0
nnpoi = 0.0
for k , j in data_dict.items():
    if j['poi'] == True:
        npoi +=1 
    else:
        nnpoi +=1

per = npoi/len(data_dict)*100.00

print "the dataset is composed of", len(data_dict), "person, with" , npoi, "person of interest."

print "The Poi in the dataset are " , per,"%"

missing = 0
for k , j in data_dict.items():
    for key , values in j.items():
        if values == "NaN":
            missing +=1 

print  "the data det conatins" , missing, " missing values." 

missingvalues = {}
for k , v in data_dict['LAY KENNETH L'].items():
    missingvalues[k]=0

for k , v in data_dict.items():
    for key , value in v.items() : 
        if value =='NaN':
            missingvalues[key] +=1 
            

print "Detailed Missing Values" 
print missingvalues


##########################################################################################

### Task 2: Remove outliers

### REMOVE OUTLIERS 

###There are 2 data point that are not person at all, so I will remove it from the dataset 

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

''' Possibile outliers that are keep in the data set. 
their value is imporant in training, most of them are poi. I tried without 
using them, once at the time and the result are worst than with them. 
data_dict.pop('LAY KENNETH L')
data_dict.pop('HIRKO JOSEPH')
data_dict.pop('SKILLING JEFFREY K')
data_dict.pop('PAI LOU L')
'''

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

### CREATE FEATURES 
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


my_dataset = data_dict


########################################################################################
'''
THIS PROCEDURE HAS BEEN USED TO CHOSE THE BEST FEATURES TO USE, 
THE RESULT IS THE LIST ABOVE 

import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier

clf= AdaBoostClassifier(learning_rate = 0.5,n_estimators=50)
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=0),
          scoring='precision')
rfecv.fit(features, labels)
print("Optimal number of features : %d" % rfecv.n_features_)
print rfecv.support_
#features=features[:,rfecv.support_]
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
'''
########################################################################################


### FEATURES SELECTED WILL BE ONLY THE FEATURES THAT HAVE LESS THAN 60 MISSING VALUES 
### AND THAN USING RFECV TO CHOSE THE BEST ONES 

features_list = ['poi',
                 'exercised_stock_options',
                 'expenses',
                 #'from_messages',
                 #'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 #'other',
                 'percent_from_poi',
                 #'percent_to_poi',
                 'restricted_stock',
                 'salary',
                 #'shared_receipt_with_poi',
                 #'to_messages',
                 #'total_payments',
                 'total_stock_value']




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



#############################################################################    

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

#############################################################################

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

########################################################################
''''

THIS PROCEDURE HAS BEEN USED TO CHOSE THE PROPER PARAMENTERS FOR THE ADABOOST ALGORTIMH
IT HAS BEEN USE GRIDSEARCH AND StratifiedShuffleSplit 

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
cv = StratifiedShuffleSplit(n_splits=15, test_size=0.20, random_state=42)
parameters = {'learning_rate':[0.5,0.7,0.8,0.9,1.0,1.1,1.2], 
                 'n_estimators':[50,75,100,150,200],
                 'algorithm' : ['SAMME.R','SAMME']}
clf = GridSearchCV(ada, parameters , cv=cv)

clf.fit(features, labels)

print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))

'''

########################################################################

clf= AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                        learning_rate=0.7, n_estimators=50, random_state=None)



########################################################################
'''

OTHERS ALGORITMH TESTED 


much faster than Adaboost but result are slightly worst
clf = GaussianNB() 

USED TO FIND THE CORRECT PARAMENTERS FOR ADABOOST 
parameters = {'learning_rate':[0.5,0.7,0.8,0.9,1.0,1.1,1.2], 
                 'n_estimators':[50,75,100,150,200]}
clf = GridSearchCV(adr, parameters)
clf=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                      learning_rate=1.0, n_estimators=75, random_state=None)


from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion ='gini',splitter='best')



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)

'''
########################################################################


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


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

from tester import test_classifier
print "Tester Classification report"
print test_classifier(clf, my_dataset, features_list)



