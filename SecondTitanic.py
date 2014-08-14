'''
Created on 12/08/2014

@author: Renato.Angrisani
'''
import pandas as pd
import numpy as np
import csv as csv
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def ageClass(nameClass, medianAge, medianYoungAge):
    
    if nameClass == 1 or nameClass == 0:
        return medianYoungAge
    else: return medianAge
    

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


def bin(value):
    
    if value > 30:
        return 3
    if value > 20:
        return 2
    if value > 10:
        return 1
    
    return 0


def TrainingFunction():
    return SVC(kernel='poly', gamma=3)
    #return KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto')


def TrainingData(trainingSet, gender, pclass):
    subsetToTrain = trainingSet[trainingSet.Pclass == pclass][trainingSet.Gender == gender]
    subsetToTrain = subsetToTrain.drop(['Gender', 'Pclass'], axis=1)
    return subsetToTrain.values


def TestingData(testingSet, gender, pclass):
    subsetToPredict = testingSet[testingSet.Pclass == pclass][testingSet.Gender == gender]
    ids = subsetToPredict['PassengerId'].values
    subsetToPredict = subsetToPredict.drop(['Gender', 'Pclass', 'PassengerId'], axis=1)
    return ids, subsetToPredict.values




train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

Ports = list(enumerate(np.unique(train_df['Embarked'])))
Ports_dict = { name : i for i, name in Ports }

if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values
    
train_df['Embarked'] = train_df['Embarked'].map( lambda x: Ports_dict[x]).astype(int)
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
train_df['Rename'] = train_df['Name'].map( lambda x: rename(x)).astype(int)
train_df['Fare'] = train_df['Fare'].map( lambda x: bin(x)).astype(int)

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

test_df['Embarked'] = test_df['Embarked'].map( lambda x: Ports_dict[x]).astype(int)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['Rename'] = test_df['Name'].map( lambda x: rename(x)).astype(int)
test_df['Fare'] = test_df['Fare'].map( lambda x: bin(x)).astype(int)

median_age = train_df['Age'].dropna().median()
median_young_age = train_df.Age[train_df.Rename < 2].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    aux = train_df.Age.isnull()
    for i in range(len(aux)):
        if aux[i]:
            train_df['Age'][i] = ageClass(train_df['Rename'][i], median_age, median_young_age)
            
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    aux = test_df.Age.isnull()
    for i in range(len(aux)):
        if aux[i]:
            test_df['Age'][i] = ageClass(test_df['Rename'][i], median_age, median_young_age)


train_df = train_df.drop(['Name', 'Sex', 'Cabin', 'Ticket', 'PassengerId', 'Parch', 'SibSp', 'Age', 'Embarked'], axis=1)
test_df = test_df.drop(['Name', 'Sex', 'Cabin', 'Ticket', 'Parch', 'SibSp', 'Age', 'Embarked'], axis=1)


dictKeys = [10, 20, 30, 11, 21, 31]
predictors = {}
predicteds = {}

predictor = TrainingFunction()

print "First class females"
train_data = TrainingData(train_df, 0, 1)
predictors[10] = predictor.fit( train_data[0::,1::], train_data[0::,0] )

ids, test_data = TestingData(test_df, 0, 1)
predicteds[10] = predictors[10].predict(test_data).astype(int)
predicteds[10] = zip(ids, predicteds[10])


print "Second class females"
train_data = TrainingData(train_df, 0, 2)
predictors[20] = predictor.fit( train_data[0::,1::], train_data[0::,0] )

ids, test_data = TestingData(test_df, 0, 2)
predicteds[20] = predictors[20].predict(test_data).astype(int)
predicteds[20] = zip(ids, predicteds[20])


print "Third class females"
train_data = TrainingData(train_df, 0, 3)
predictors[30] = predictor.fit( train_data[0::,1::], train_data[0::,0] )

ids, test_data = TestingData(test_df, 0, 3)
predicteds[30] = predictors[30].predict(test_data).astype(int)
predicteds[30] = zip(ids, predicteds[30])


print "First class males"
train_data = TrainingData(train_df, 1, 1)
predictors[11] = predictor.fit( train_data[0::,1::], train_data[0::,0] )

ids, test_data = TestingData(test_df, 1, 1)
predicteds[11] = predictors[11].predict(test_data).astype(int)
predicteds[11] = zip(ids, predicteds[11])


print "Second class males"
train_data = TrainingData(train_df, 1, 2)
predictors[21] = predictor.fit( train_data[0::,1::], train_data[0::,0] )

ids, test_data = TestingData(test_df, 1, 2)
predicteds[21] = predictors[21].predict(test_data).astype(int)
predicteds[21] = zip(ids, predicteds[21])


print "Third class males"
train_data = TrainingData(train_df, 1, 3)
predictors[31] = predictor.fit( train_data[0::,1::], train_data[0::,0] )

ids, test_data = TestingData(test_df, 1, 3)
predicteds[31] = predictors[31].predict(test_data).astype(int)
predicteds[31] = zip(ids, predicteds[31])




predictions = [x for k in dictKeys for x in predicteds[k]]
predictions = sorted(predictions, key=lambda tup: tup[0])

predictions_file = open("mysecondsvm.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(predictions)
predictions_file.close()
print "Done!"





