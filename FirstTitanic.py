""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014

Modified by: Renato Giaffredo Angrisani
Modified on: 12th August 2014

Included SVM and Logistic Regression

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def bin(value):
    
    if value > 30:
        return 3
    if value > 20:
        return 2
    if value > 10:
        return 1
    
    return 0

#def rename(value):
#    
#    if "Master." in value:
#        return 0
#    if "Miss." in value:
#        return 1
#    if "Mr." in value or "Dr." in value or "Rev." in value or "Col." in value:
#        return 2
#    if "Don." in value or "Sir." in value or "Major.":
#        return 3
#    if "Mrs." in value or "Ms." in value:
#        return 4
#    if "Lady." in value or "Mme." in value:
#        return 5
#    
#    return 2

def rename(value):
    
    if "Master." in value:
        return 0
    if "Miss." in value:
        return 0
    if "Mr." in value or "Dr." in value or "Rev." in value or "Col." in value:
        return 1
    if "Don." in value or "Sir." in value or "Major.":
        return 2
    if "Mrs." in value or "Ms." in value:
        return 2
    if "Lady." in value or "Mme." in value:
        return 2
    
    return 2

def binarize(value):
    
    if value > 0:
        return 1
    else:
        return 0
    
def ageClass(nameClass, parch, medianAge):
    
    if (nameClass == 1 or nameClass == 0) and parch > 0:
        return 5
    else: return medianAge
    
def CabinBin(value):
    
    if type(value) is float:
        return 0
    else:
        return 1
    

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
train_df['Rename'] = train_df['Name'].map( lambda x: rename(x)).astype(int)
train_df['SibSp'] = train_df['SibSp'].map( lambda x: binarize(x)).astype(int)
train_df['Parch'] = train_df['Parch'].map( lambda x: binarize(x)).astype(int)
train_df['Cabin'] = train_df['Cabin'].map( lambda x: CabinBin(x)).astype(int)
train_df['Fare'] = train_df['Fare'].map( lambda x: bin(x)).astype(int)


# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int


# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    aux = train_df.Age.isnull()
    for i in range(len(aux)):
        if aux[i]:
            train_df['Age'][i] = ageClass(train_df['Rename'][i], train_df['Parch'][i], median_age)
            
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Cabin', 'Ticket', 'PassengerId', 'Age', 'Parch', 'SibSp', 'Embarked'], axis=1) 

# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['Rename'] = test_df['Name'].map( lambda x: rename(x)).astype(int)
test_df['SibSp'] = test_df['SibSp'].map( lambda x: binarize(x)).astype(int)
test_df['Parch'] = test_df['Parch'].map( lambda x: binarize(x)).astype(int)
test_df['Cabin'] = test_df['Cabin'].map( lambda x: CabinBin(x)).astype(int)
test_df['Fare'] = test_df['Fare'].map( lambda x: bin(x)).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    aux = test_df.Age.isnull()
    for i in range(len(aux)):
        if aux[i]:
            test_df['Age'][i] = ageClass(test_df['Rename'][i], test_df['Parch'][i], median_age)
#if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
#    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Cabin', 'Ticket', 'PassengerId', 'Age', 'Parch', 'SibSp', 'Embarked'], axis=1)


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training Random Forest...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting Random Forest...'
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done Random Forest.'

print 'Training SVM...'
clf = SVC(kernel='poly', gamma=3)
clf = clf.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting SVM...'
output = clf.predict(test_data).astype(int)


predictions_file = open("myfirstsvm.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done SVM.'

print 'Training LR...'
clf = LogisticRegression(penalty='l1', dual=False, C=1.0)
clf = clf.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting LR...'
output = clf.predict(test_data).astype(int)


predictions_file = open("myfirstlr.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done LR.'

