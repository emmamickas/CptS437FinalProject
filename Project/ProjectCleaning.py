#Emma Mickas
#ID: 011651116
#CptS437 Project

import csv
from distutils.command.clean import clean
import math
import numpy as np
import numpy.ma as ma
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.linear_model import Perceptron
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
from scipy import ndimage
from time import time
from sklearn import manifold, datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.fixes import parse_version
from sklearn.feature_extraction.image import grid_to_graph
import skimage
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn.svm import l1_min_c
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import metrics

allUsersQuestions = {}
allQuestions = {}
allQuestionNames = []

def CleanDataForReading():
    open_file = open('data-final.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Filtered.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 1

    for line in file_in:
        if i == 1:
            i += 1
        if 'NULL' not in line[:50]:
            file_out.writerow(line[:50])
    
    open_file.close()
    open_file2.close()

def TestingShrinker():
    open_file = open('data-final - Filtered.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Testing.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 0

    for line in file_in:
        if i == 0:
            i += 1
            continue
        file_out.writerow(line[:100])
        i += 1
        if i > 10:
            break
    
    open_file.close()
    open_file2.close()

    open_file = open('data-final - Testing.csv', 'r', encoding='utf-8', newline='')
    fulldataset = np.loadtxt(open_file, delimiter='\t', usecols=(range(100)), max_rows=5000)

    print(fulldataset.shape)

    open_file.close()

    open_file = open('data-final - Testing.csv', 'r', encoding='utf-8', newline='')
    dataset = np.loadtxt(open_file, delimiter='\t', usecols=(range(50)), max_rows=5000)

    print(dataset.shape)

    open_file.close()

    trainingdataset = dataset[0:7]
    testdataset = dataset[7:10]

    return dataset, trainingdataset, testdataset

def Shrinker():
    open_file = open('data-final - Filtered.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Shrunken.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 0
    linecount = 0
    start = 0

    for line in file_in:
        if start == 0:
            start += 1
            continue
        file_out.writerow(line[:100])
        linecount += 1
        i += 1
        if linecount >= 10:
            break
    
    open_file.close()
    open_file2.close()
    return

def ReadInData():

    #open_file = open('data-final - Shrunken.csv', 'r', encoding='utf-8', newline='') # Shrunken version of the dataset for ease of testing
    open_file = open('data-final - Filtered.csv', 'r', encoding='utf-8', newline='') # Full dataset after first cleaning (filtering out unnecessary values)
    fulldataset = np.loadtxt(open_file, delimiter='\t', usecols=(range(50)), max_rows=1015000, skiprows=1) # Read in all the data, skip the header

    print(fulldataset.shape)

    fulllength, throwaway = fulldataset.shape
    #split = int(fulllength) * 0.67 # Manual split method, makes testing a bit easier
    #sampledataset = fulldataset[0:int(split)] # Manual split method, makes testing a bit easier
    #testdataset = fulldataset[int(split):int(fulllength)] # Manual split method, makes testing a bit easier

    open_file.close()

    dataset = resample(fulldataset, n_samples=20, random_state=1, replace=False) # Get only 10000 samples
    sampledataset, testdataset = train_test_split(dataset, test_size=0.33, random_state=1) # Split into training and testing

    print("Done reading in data...")

    return dataset, sampledataset, testdataset

def CleanDataPart1(dataset):
    cleaneddataset = np.zeros(shape=(dataset.shape[0], 50))

    i = 0

    # Scale each value so that more positive means more fitting the personality trait

    for row in dataset:
        for j in range(50):
            if j >= 0 and j < 10: # Extroversion question
                if (j % 2) == 0: # Even numbered question, higher answer = more extroverted
                    cleaneddataset[i][j] = row[j]
                else: # Odd numbered question, higher answer = less extroverted
                    cleaneddataset[i][j] = 5 - row[j]
            elif j >= 10 and j < 20: # Neuroticism question
                if j == 10:
                    cleaneddataset[i][j] = row[j]
                elif j == 11:
                    cleaneddataset[i][j] = 5 - row[j]
                elif j == 12:
                    cleaneddataset[i][j] = row[j]
                elif j == 13:
                    cleaneddataset[i][j] = 5 - row[j]
                else:
                    cleaneddataset[i][j] = row[j]
            elif  j >= 20 and j < 30: # Agreeableness question
                if (j % 2) == 0: # Even numbered question, higher answer = less agreeable
                    cleaneddataset[i][j] = 5 - row[j]
                else: # Odd numbered question, higher answer = more agreeable
                    cleaneddataset[i][j] = row[j]
            elif j >= 30 and j < 40: # Conscientiousness question
                if (j % 2) == 0: # Even numbered question, higher answer = more conscientious
                    cleaneddataset[i][j] = row[j]
                else: # Odd numbered question, higher answer = less conscientious
                    cleaneddataset[i][j] = 5 - row[j]
            elif j >= 40 and j < 50: # Openness question
                if (j == 47):
                    cleaneddataset[i][j] = row[j]
                elif (j == 48):
                    cleaneddataset[i][j] = 5 - row[j]
                elif (j == 49):
                    cleaneddataset[i][j] = row[j]
                elif (j % 2) == 0: # Even numbered question, higher answer = more open
                    cleaneddataset[i][j] = row[j]
                else: # Odd numbered question, higher answer = less open
                    cleaneddataset[i][j] = 5 - row[j]

        i += 1

    return cleaneddataset

def NormalizeData(dataset):
    normalized_array = normalize(dataset, norm="l1")
    return normalized_array

def CalculateExtroversion(row):
    total = 0
    total += row[0]
    total -= row[1]
    total += row[2]
    total -= row[3]
    total += row[4]
    total -= row[5]
    total += row[6]
    total -= row[7]
    total += row[8]
    total -= row[9]
    return total

def CalculateNeuroticism(row):
    total = 0
    total += row[10]
    total -= row[11]
    total += row[12]
    total -= row[13]
    total += row[14]
    total += row[15]
    total += row[16]
    total += row[17]
    total += row[18]
    total += row[19]
    return total

def CalculateAgreeableness(row):
    total = 0
    total -= row[20]
    total += row[21]
    total -= row[22]
    total += row[23]
    total -= row[24]
    total += row[25]
    total -= row[26]
    total += row[27]
    total -= row[28]
    total += row[29]
    return total

def CalculateConscientiousness(row):
    total = 0
    total += row[30]
    total -= row[31]
    total += row[32]
    total -= row[33]
    total += row[34]
    total -= row[35]
    total += row[36]
    total -= row[37]
    total += row[38]
    total -= row[39]
    return total

def CalculateOpenness(row):
    total = 0
    total += row[40]
    total -= row[41]
    total += row[42]
    total -= row[43]
    total += row[44]
    total -= row[45]
    total += row[46]
    total += row[47]
    total -= row[48]
    total += row[49]
    return total

def CalculateCleanedTotals(dataset):
    datasettotals = np.zeros(shape=(dataset.shape[0], 5))

    i = 0

    for row in dataset:
        for j in range(0, 10):
            datasettotals[i][0] += row[j]
        for j in range(10, 20):
            datasettotals[i][1] += row[j]
        for j in range(20, 30):
            datasettotals[i][2] += row[j]
        for j in range(30, 40):
            datasettotals[i][3] += row[j]
        for j in range(40, 50):
            datasettotals[i][4] += row[j]
        i += 1

    return datasettotals

def CalculateIndividualTotals(dataset):
    open_file = open('data-final - IndividualTotals.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    datasettotals = np.zeros(shape=(dataset.shape[0], 5))

    i = 0

    for row in dataset:
        datasettotals[i][0] = CalculateExtroversion(row)
        datasettotals[i][1] = CalculateNeuroticism(row)
        datasettotals[i][2] = CalculateAgreeableness(row)
        datasettotals[i][3] = CalculateConscientiousness(row)
        datasettotals[i][4] = CalculateOpenness(row)
        file_out.writerow(datasettotals[i])
        i += 1

    open_file.close()

    return datasettotals

def CalculateIndividualPreferences(dataset):
    datasetpreferences = np.zeros(shape=(dataset.shape[0], 5))

    i = 0

    for row in dataset:
        if row[0] >= 0:
            datasetpreferences[i][0] = 1
        elif row[0] < 0:
            datasetpreferences[i][0] = -1

        if row[1] >= 0:
            datasetpreferences[i][1] = 1
        elif row[1] < 0:
            datasetpreferences[i][1] = -1

        if row[2] >= 0:
            datasetpreferences[i][2] = 1
        elif row[2] < 0:
            datasetpreferences[i][2] = -1

        if row[3] >= 0:
            datasetpreferences[i][3] = 1
        elif row[3] < 0:
            datasetpreferences[i][3] = -1

        if row[4] >= 0:
            datasetpreferences[i][4] = 1
        elif row[4] < 0:
            datasetpreferences[i][4] = -1

        i += 1

    return datasetpreferences

def CalculateCleanedPreferences(datasettotals):
    datasetpreferences = np.zeros(shape=(datasettotals.shape[0], 5))

    i = 0

    for row in datasettotals:
        if row[0] >= 25:
            datasetpreferences[i][0] = 1
        elif row[0] < 25:
            datasetpreferences[i][0] = -1

        if row[1] >= 25:
            datasetpreferences[i][1] = 1
        elif row[1] < 25:
            datasetpreferences[i][1] = -1

        if row[2] >= 25:
            datasetpreferences[i][2] = 1
        elif row[2] < 25:
            datasetpreferences[i][2] = -1

        if row[3] >= 25:
            datasetpreferences[i][3] = 1
        elif row[3] < 25:
            datasetpreferences[i][3] = -1

        if row[4] >= 25:
            datasetpreferences[i][4] = 1
        elif row[4] < 25:
            datasetpreferences[i][4] = -1

        i += 1

    return datasetpreferences

def PerceptronForPruning(dataset, datasetpreferences, characteristicstoprune, characteristicstopredict, iterations, learningrate, numtopruneto):
    weight = [0] * 10
    weightSums = [0] * 10

    mistakes = 0
    iterationMistakeTotals = [0] * iterations
    iterationAccuracies = []
    count = 0

    datasetsamples = dataset[:,characteristicstoprune[0]:characteristicstoprune[1]]

    datasetlabels = datasetpreferences[:,characteristicstopredict[0]]

    file_out = open('perceptronoutput.txt', 'a')

    for iteration in range(iterations):

        iterationAccuracies.append([0,0.0]) # Add a new element to keep track of iteration mistake count and iteration accuracy percent
        sampleNumber = 0 # Reset example number for the next iteration

        #print("Iteration:", iteration)

        for sample in datasetsamples:
            #print("Sample:", sample)
            #print("Current label:", datasetlabels[sampleNumber])
            #print("Prediction:", np.dot(weight, sample))

            if (datasetlabels[sampleNumber]) * (np.dot(weight, sample)) <= 0: # Mistake!
                mistakes += 1 # Increment total mistakes
                iterationMistakeTotals[iteration] += 1 # Increment mistakes for the current iteration
                weight = weight + np.multiply((learningrate * datasetlabels[sampleNumber]), sample) # Update weight
                #print("Weights:", weight)
                weightSums = np.add(weightSums, weight) # Update weight total to calculate average weight later
                #weightSums = weightSums + weight
                count += 1 # Keep track of total number of weights to calculate average weight later

            sampleNumber += 1

        #iterationAccuracies[iteration][0] = PerceptronTesting(weight, datasetsamples, examplesDictionary, len(examplesDictionary)) # Save this iteration's accuracy on training data
        #iterationAccuracies[iteration][1] = PerceptronTesting(weight, TestingExamples, testingDictionary, len(testingDictionary)) # Save this iterations's accuracy on testing data

    #for iteration in range(len(iterationMistakeTotals)): # Write iteration mistake counts to file
        #print("iteration-{} {}".format(iteration + 1, iterationMistakeTotals[iteration]), file=file_out)

    #print("", file=file_out)

    #for iteration in range(len(iterationAccuracies)): # Write iteration accuracies to file
        #print("iteration-{} {} {}".format(iteration + 1, iterationAccuracies[iteration][0], iterationAccuracies[iteration][1]), file=file_out)

    averageWeights = np.divide(weightSums, count)

    file_out.close()

    maxWeightIndexes = [-1] * numtopruneto

    for i in range(numtopruneto):
        curMaxWeight = -999999
        curMaxWeightIndex = -1
        for j in range(len(weight)):
            #print("curMaxWeight:", curMaxWeight)
            #print("abs(weight[j]):", abs(weight[j]))
            if j in maxWeightIndexes:
                continue
            if (abs(weight[j]) > curMaxWeight):
                curMaxWeight = abs(weight[j])
                curMaxWeightIndex = j
        maxWeightIndexes[i] = curMaxWeightIndex

    print("maxWeightIndexes:", maxWeightIndexes)

    selectedweights = [0] * numtopruneto

    for i in range(numtopruneto):
        selectedweights[i] = weight[maxWeightIndexes[i]]

    print("Selectedweights:", selectedweights)

    return maxWeightIndexes, weight, averageWeights

def PredictExtroversion(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    #print("alltrainingdatasetquestions:", alltrainingdatasetquestions)
    #print("alltrainingdatasetquestions[:,10:]:", alltrainingdatasetquestions[:,10:])
    #print("allsampledatasettotals:", allsampledatasettotals)
    #print("allsampledatasetpreferences:", allsampledatasetpreferences)
    #print("alltestdatasettotals:", alltestdatasettotals)
    #print("alltestdatasetpreferences:", alltestdatasetpreferences)
    #print("allsampledatasettotals[:,1:]: ", allsampledatasettotals[:,1:])
    #print("allsampledatasetpreferences[:,1:]:", allsampledatasetpreferences[:,1:])
    #print("allsampledatasetpreferences[:,0]:", allsampledatasetpreferences[:,0])

    trainingdatasetquestions = alltrainingdatasetquestions[:,10:] # Select all question columns not pertaining to extroversion
    trainingdatasettotals = alltrainingdatasettotals[:,1:] # Select all total columns but extroversion
    trainingdatasetpreferences = alltrainingdatasetpreferences[:,1:] # Select all preference columns but extroversion
    trainingcorrectlabels = alltrainingdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns

    trainingcorrectlabels = np.transpose(trainingcorrectlabels)

    #print("correctlabels:", correctlabels)
    
    testingdatasetquestions = alltestingdatasetquestions[:,10:] # Select all question columns not pertaining to extroversion
    testingdatasettotals = alltestingdatasettotals[:,1:] # Select all columns but extroversion
    testingdatasetpreferences = alltestingdatasetpreferences[:,1:] # Select all preference columns but extroversion
    testingcorrectlabels = alltestingdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)

    #print("correctlabels:", trainingcorrectlabels)
    #print("testcorrectlabels:", testingcorrectlabels)

    #print("\n")


    # print("Perceptron:")
    # print("perceptquestionpredictions:", perceptquestionpredictions)
    # print("testquestionpredictions:", testquestionpredictions)
    # print("perceptpredictions:", perceptpredictions)
    # print("testpredictions:", testpredictions)
    # print("preferencepredictions:", preferencepredictions)
    # print("testpreferencepredictions:", testpreferencepredictions)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict extroversion based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict extroversion based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict extroversion based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict extroversion based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict extroversion based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    
    # print("SGD:")
    # print("sgdquestionpredictions:", sgdquestionpredictions)
    # print("sgdtestquestionpredictions:", sgdtestquestionpredictions)
    # print("sgdpredictions:", sgdpredictions)
    # print("sgdtestpredictions:", sgdtestpredictions)
    # print("sgdpreferencepredictions:", sgdpreferencepredictions)
    # print("sgdtestpreferencepredictions:", sgdtestpreferencepredictions)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict extroversion based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict extroversion based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict extroversion based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict extroversion based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict extroversion based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)


    # print("Logistic:")
    # print("logisticquestionpredictions:", logisticquestionpredictions)
    # print("logistictestquestionpredictions:", logistictestquestionpredictions)
    # print("logisticpredictions:", logisticpredictions)
    # print("logistictestpredictions:", logistictestpredictions)
    # print("logisticpreferencepredictions:", logisticpreferencepredictions)
    # print("logistictestpreferencepredictions:", logistictestpreferencepredictions)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict extroversion based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict extroversion based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict extroversion based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict extroversion based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict extroversion based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    
    # print("DT:")
    # print("decisiontreequestionpredictions:", decisiontreequestionpredictions)
    # print("decisiontreetestquestionpredictions:", decisiontreetestquestionpredictions)
    # print("decisiontreepredictions:", decisiontreepredictions)
    # print("decisiontreetestpredictions:", decisiontreetestpredictions)
    # print("decisiontreepreferencepredictions:", decisiontreepreferencepredictions)
    # print("decisiontreetestpreferencepredictions:", decisiontreetestpreferencepredictions)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict extroversion based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict extroversion based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict extroversion based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict extroversion based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict extroversion based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict extroversion based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict extroversion based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict extroversion based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    return

def PredictNeuroticism(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    columnstodelete = list(range(10, 20))

    # print("alltrainingdatasetquestions:", alltrainingdatasetquestions)
    # print("np.delete(alltrainingdatasetquestions, columnstodelete, axis=1):", np.delete(alltrainingdatasetquestions, columnstodelete, axis=1))
    # print("alltrainingdatasettotals:", alltrainingdatasettotals)
    # print("alltrainingdatasetpreferences:", alltrainingdatasetpreferences)
    # print("alltestingdatasettotals:", alltestingdatasettotals)
    # print("alltestingdatasetpreferences:", alltestingdatasetpreferences)
    # print("np.delete(alltrainingdatasettotals, 1, axis=1): ", np.delete(alltrainingdatasettotals, 1, axis=1))
    # print("np.delete(alltrainingdatasetpreferences, 1, axis=1):", np.delete(alltrainingdatasetpreferences, 1, axis=1))
    # print("alltrainingdatasetpreferences[:,1]:", alltrainingdatasetpreferences[:,1])

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 1, axis=1) # Select all columns but neuroticism
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 1, axis=1) # Select all columns but neuroticism
    trainingcorrectlabels = alltrainingdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    
    #print("correctlabels:", correctlabels)

    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    testingdatasettotals = np.delete(alltestingdatasettotals, 1, axis=1) # Select all columns but neuroticism
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 1, axis=1) # Select all columns but neuroticism
    testingcorrectlabels = alltestingdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)

    #print("correctlabels:", trainingcorrectlabels)
    #print("testcorrectlabels:", testingcorrectlabels)

    #print("\n")


    # print("Perceptron:")
    # print("perceptquestionpredictions:", perceptquestionpredictions)
    # print("testquestionpredictions:", testquestionpredictions)
    # print("perceptpredictions:", perceptpredictions)
    # print("testpredictions:", testpredictions)
    # print("preferencepredictions:", preferencepredictions)
    # print("testpreferencepredictions:", testpreferencepredictions)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict neuroticism based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict neuroticism based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    
    # print("SGD:")
    # print("sgdquestionpredictions:", sgdquestionpredictions)
    # print("sgdtestquestionpredictions:", sgdtestquestionpredictions)
    # print("sgdpredictions:", sgdpredictions)
    # print("sgdtestpredictions:", sgdtestpredictions)
    # print("sgdpreferencepredictions:", sgdpreferencepredictions)
    # print("sgdtestpreferencepredictions:", sgdtestpreferencepredictions)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict neuroticism based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict neuroticism based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)


    # print("Logistic:")
    # print("logisticquestionpredictions:", logisticquestionpredictions)
    # print("logistictestquestionpredictions:", logistictestquestionpredictions)
    # print("logisticpredictions:", logisticpredictions)
    # print("logistictestpredictions:", logistictestpredictions)
    # print("logisticpreferencepredictions:", logisticpreferencepredictions)
    # print("logistictestpreferencepredictions:", logistictestpreferencepredictions)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict neuroticism based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict neuroticism based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    
    # print("DT:")
    # print("decisiontreequestionpredictions:", decisiontreequestionpredictions)
    # print("decisiontreetestquestionpredictions:", decisiontreetestquestionpredictions)
    # print("decisiontreepredictions:", decisiontreepredictions)
    # print("decisiontreetestpredictions:", decisiontreetestpredictions)
    # print("decisiontreepreferencepredictions:", decisiontreepreferencepredictions)
    # print("decisiontreetestpreferencepredictions:", decisiontreetestpreferencepredictions)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict neuroticism based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict neuroticism based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict neuroticism based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    return

def PredictAgreeableness(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    columnstodelete = list(range(20, 30))

    # print("alltrainingdatasetquestions:", alltrainingdatasetquestions)
    # print("np.delete(alltrainingdatasetquestions, columnstodelete, axis=1):", np.delete(alltrainingdatasetquestions, columnstodelete, axis=1))
    # print("alltrainingdatasettotals:", alltrainingdatasettotals)
    # print("alltrainingdatasetpreferences:", alltrainingdatasetpreferences)
    # print("alltestingdatasettotals:", alltestingdatasettotals)
    # print("alltestingdatasetpreferences:", alltestingdatasetpreferences)
    # print("np.delete(alltrainingdatasettotals, 1, axis=1): ", np.delete(alltrainingdatasettotals, 1, axis=1))
    # print("np.delete(alltrainingdatasetpreferences, 1, axis=1):", np.delete(alltrainingdatasetpreferences, 1, axis=1))
    # print("alltrainingdatasetpreferences[:,1]:", alltrainingdatasetpreferences[:,1])

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 2, axis=1) # Select all columns but agreeableness
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 2, axis=1) # Select all columns but agreeableness
    trainingcorrectlabels = alltrainingdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    
    #print("correctlabels:", correctlabels)

    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    testingdatasettotals = np.delete(alltestingdatasettotals, 2, axis=1) # Select all columns but agreeableness
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 2, axis=1) # Select all columns but agreeableness
    testingcorrectlabels = alltestingdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)

    #print("correctlabels:", trainingcorrectlabels)
    #print("testcorrectlabels:", testingcorrectlabels)

    #print("\n")


    # print("Perceptron:")
    # print("perceptquestionpredictions:", perceptquestionpredictions)
    # print("testquestionpredictions:", testquestionpredictions)
    # print("perceptpredictions:", perceptpredictions)
    # print("testpredictions:", testpredictions)
    # print("preferencepredictions:", preferencepredictions)
    # print("testpreferencepredictions:", testpreferencepredictions)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict agreeableness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict agreeableness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    
    # print("SGD:")
    # print("sgdquestionpredictions:", sgdquestionpredictions)
    # print("sgdtestquestionpredictions:", sgdtestquestionpredictions)
    # print("sgdpredictions:", sgdpredictions)
    # print("sgdtestpredictions:", sgdtestpredictions)
    # print("sgdpreferencepredictions:", sgdpreferencepredictions)
    # print("sgdtestpreferencepredictions:", sgdtestpreferencepredictions)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict agreeableness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict agreeableness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)


    # print("Logistic:")
    # print("logisticquestionpredictions:", logisticquestionpredictions)
    # print("logistictestquestionpredictions:", logistictestquestionpredictions)
    # print("logisticpredictions:", logisticpredictions)
    # print("logistictestpredictions:", logistictestpredictions)
    # print("logisticpreferencepredictions:", logisticpreferencepredictions)
    # print("logistictestpreferencepredictions:", logistictestpreferencepredictions)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict agreeableness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict agreeableness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    
    # print("DT:")
    # print("decisiontreequestionpredictions:", decisiontreequestionpredictions)
    # print("decisiontreetestquestionpredictions:", decisiontreetestquestionpredictions)
    # print("decisiontreepredictions:", decisiontreepredictions)
    # print("decisiontreetestpredictions:", decisiontreetestpredictions)
    # print("decisiontreepreferencepredictions:", decisiontreepreferencepredictions)
    # print("decisiontreetestpreferencepredictions:", decisiontreetestpreferencepredictions)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict agreeableness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict agreeableness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict agreeableness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    return

def PredictConscientiousness(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    columnstodelete = list(range(30, 40))

    # print("alltrainingdatasetquestions:", alltrainingdatasetquestions)
    # print("np.delete(alltrainingdatasetquestions, columnstodelete, axis=1):", np.delete(alltrainingdatasetquestions, columnstodelete, axis=1))
    # print("alltrainingdatasettotals:", alltrainingdatasettotals)
    # print("alltrainingdatasetpreferences:", alltrainingdatasetpreferences)
    # print("alltestingdatasettotals:", alltestingdatasettotals)
    # print("alltestingdatasetpreferences:", alltestingdatasetpreferences)
    # print("np.delete(alltrainingdatasettotals, 1, axis=1): ", np.delete(alltrainingdatasettotals, 1, axis=1))
    # print("np.delete(alltrainingdatasetpreferences, 1, axis=1):", np.delete(alltrainingdatasetpreferences, 1, axis=1))
    # print("alltrainingdatasetpreferences[:,1]:", alltrainingdatasetpreferences[:,1])

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 3, axis=1) # Select all columns but conscientiousness
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 3, axis=1) # Select all columns but conscientiousness
    trainingcorrectlabels = alltrainingdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    
    #print("correctlabels:", correctlabels)

    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    testingdatasettotals = np.delete(alltestingdatasettotals, 3, axis=1) # Select all columns but conscientiousness
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 3, axis=1) # Select all columns but conscientiousness
    testingcorrectlabels = alltestingdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)

    #print("correctlabels:", trainingcorrectlabels)
    #print("testcorrectlabels:", testingcorrectlabels)

    #print("\n")


    # print("Perceptron:")
    # print("perceptquestionpredictions:", perceptquestionpredictions)
    # print("testquestionpredictions:", testquestionpredictions)
    # print("perceptpredictions:", perceptpredictions)
    # print("testpredictions:", testpredictions)
    # print("preferencepredictions:", preferencepredictions)
    # print("testpreferencepredictions:", testpreferencepredictions)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict conscientiousness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict conscientiousness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    
    # print("SGD:")
    # print("sgdquestionpredictions:", sgdquestionpredictions)
    # print("sgdtestquestionpredictions:", sgdtestquestionpredictions)
    # print("sgdpredictions:", sgdpredictions)
    # print("sgdtestpredictions:", sgdtestpredictions)
    # print("sgdpreferencepredictions:", sgdpreferencepredictions)
    # print("sgdtestpreferencepredictions:", sgdtestpreferencepredictions)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict conscientiousness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict conscientiousness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)


    # print("Logistic:")
    # print("logisticquestionpredictions:", logisticquestionpredictions)
    # print("logistictestquestionpredictions:", logistictestquestionpredictions)
    # print("logisticpredictions:", logisticpredictions)
    # print("logistictestpredictions:", logistictestpredictions)
    # print("logisticpreferencepredictions:", logisticpreferencepredictions)
    # print("logistictestpreferencepredictions:", logistictestpreferencepredictions)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict conscientiousness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict conscientiousness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    
    # print("DT:")
    # print("decisiontreequestionpredictions:", decisiontreequestionpredictions)
    # print("decisiontreetestquestionpredictions:", decisiontreetestquestionpredictions)
    # print("decisiontreepredictions:", decisiontreepredictions)
    # print("decisiontreetestpredictions:", decisiontreetestpredictions)
    # print("decisiontreepreferencepredictions:", decisiontreepreferencepredictions)
    # print("decisiontreetestpreferencepredictions:", decisiontreetestpreferencepredictions)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict conscientiousness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict conscientiousness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    return

def PredictOpenness(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    columnstodelete = list(range(40, 50))

    # print("alltrainingdatasetquestions:", alltrainingdatasetquestions)
    # print("np.delete(alltrainingdatasetquestions, columnstodelete, axis=1):", np.delete(alltrainingdatasetquestions, columnstodelete, axis=1))
    # print("alltrainingdatasettotals:", alltrainingdatasettotals)
    # print("alltrainingdatasetpreferences:", alltrainingdatasetpreferences)
    # print("alltestingdatasettotals:", alltestingdatasettotals)
    # print("alltestingdatasetpreferences:", alltestingdatasetpreferences)
    # print("np.delete(alltrainingdatasettotals, 1, axis=1): ", np.delete(alltrainingdatasettotals, 1, axis=1))
    # print("np.delete(alltrainingdatasetpreferences, 1, axis=1):", np.delete(alltrainingdatasetpreferences, 1, axis=1))
    # print("alltrainingdatasetpreferences[:,1]:", alltrainingdatasetpreferences[:,1])

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 4, axis=1) # Select all columns but openness
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 4, axis=1) # Select all columns but openness
    trainingcorrectlabels = alltrainingdatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    #print("correctlabels:", correctlabels)

    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    testingdatasettotals = np.delete(alltestingdatasettotals, 4, axis=1) # Select all columns but openness
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 4, axis=1) # Select all columns but openness
    testingcorrectlabels = alltestingdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)

    #print("correctlabels:", trainingcorrectlabels)
    #print("testcorrectlabels:", testingcorrectlabels)

    #print("\n")


    # print("Perceptron:")
    # print("perceptquestionpredictions:", perceptquestionpredictions)
    # print("testquestionpredictions:", testquestionpredictions)
    # print("perceptpredictions:", perceptpredictions)
    # print("testpredictions:", testpredictions)
    # print("preferencepredictions:", preferencepredictions)
    # print("testpreferencepredictions:", testpreferencepredictions)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict openness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict openness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict openness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict openness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict openness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict openness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict openness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict openness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict openness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict openness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    
    # print("SGD:")
    # print("sgdquestionpredictions:", sgdquestionpredictions)
    # print("sgdtestquestionpredictions:", sgdtestquestionpredictions)
    # print("sgdpredictions:", sgdpredictions)
    # print("sgdtestpredictions:", sgdtestpredictions)
    # print("sgdpreferencepredictions:", sgdpreferencepredictions)
    # print("sgdtestpreferencepredictions:", sgdtestpreferencepredictions)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict openness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict openness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict openness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict openness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict openness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict openness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict openness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict openness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict openness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict openness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)


    # print("Logistic:")
    # print("logisticquestionpredictions:", logisticquestionpredictions)
    # print("logistictestquestionpredictions:", logistictestquestionpredictions)
    # print("logisticpredictions:", logisticpredictions)
    # print("logistictestpredictions:", logistictestpredictions)
    # print("logisticpreferencepredictions:", logisticpreferencepredictions)
    # print("logistictestpreferencepredictions:", logistictestpreferencepredictions)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict openness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict openness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict openness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict openness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict openness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict openness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict openness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict openness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict openness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict openness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    
    # print("DT:")
    # print("decisiontreequestionpredictions:", decisiontreequestionpredictions)
    # print("decisiontreetestquestionpredictions:", decisiontreetestquestionpredictions)
    # print("decisiontreepredictions:", decisiontreepredictions)
    # print("decisiontreetestpredictions:", decisiontreetestpredictions)
    # print("decisiontreepreferencepredictions:", decisiontreepreferencepredictions)
    # print("decisiontreetestpreferencepredictions:", decisiontreetestpreferencepredictions)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict openness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict openness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict openness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict openness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict openness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict openness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict openness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict openness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict openness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict openness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    return

def PredictBasedOnExtroversion(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    extroversionquestions = allsampledataset[:,0:10] # Select extroversion question columns
    print(extroversionquestions)
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    print(neuroticismcorrectlabels)
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testextroversionquestions = alltestdataset[:,0:10] # Select extroversion question columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns


    # NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(extroversionquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(extroversionquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testextroversionquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(extroversionquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testextroversionquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(extroversionquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testextroversionquestions)
    
    neuroticismclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(extroversionquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testextroversionquestions)

    # Evaluation
    neuroticism_perceptron_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions)

    print("Able to predict neuroticism based on extroversion training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy), file=file_out)

    neuroticism_perceptron_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions)

    print("Able to predict neuroticism based on extroversion testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy), file=file_out)

    neuroticism_sgd_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions)

    print("Able to predict neuroticism based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy), file=file_out)

    neuroticism_sgd_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions)

    print("Able to predict neuroticism based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy), file=file_out)

    neuroticism_logistic_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions)

    print("Able to predict neuroticism based on extroversion training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy), file=file_out)

    neuroticism_logistic_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions)

    print("Able to predict neuroticism based on extroversion testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy), file=file_out)

    neuroticism_dt_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions)

    print("Able to predict neuroticism based on extroversion training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy), file=file_out)

    neuroticism_dt_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions)

    print("Able to predict neuroticism based on extroversion testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy), file=file_out)

    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(extroversionquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(extroversionquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testextroversionquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(extroversionquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testextroversionquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(extroversionquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testextroversionquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(extroversionquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testextroversionquestions)

    # Evaluation
    agreeableness_perceptron_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions)

    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy), file=file_out)

    agreeableness_perceptron_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy), file=file_out)

    agreeableness_sgd_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions)

    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy), file=file_out)

    agreeableness_sgd_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy), file=file_out)

    agreeableness_logistic_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions)

    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy), file=file_out)

    agreeableness_logistic_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy), file=file_out)

    agreeableness_dt_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions)

    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy), file=file_out)

    agreeableness_dt_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy), file=file_out)

    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(extroversionquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(extroversionquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testextroversionquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(extroversionquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testextroversionquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(extroversionquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testextroversionquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(extroversionquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testextroversionquestions)

    # Evaluation
    conscientiousness_perceptron_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions)

    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy), file=file_out)

    conscientiousness_perceptron_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy), file=file_out)

    conscientiousness_sgd_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions)

    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy), file=file_out)

    conscientiousness_sgd_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy), file=file_out)

    conscientiousness_logistic_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions)

    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy), file=file_out)

    conscientiousness_logistic_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy), file=file_out)

    conscientiousness_dt_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions)

    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy), file=file_out)

    conscientiousness_dt_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy), file=file_out)

    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3.fit(extroversionquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(extroversionquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testextroversionquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd.fit(extroversionquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(extroversionquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testextroversionquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l2', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(extroversionquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(extroversionquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testextroversionquestions)
    
    opennessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree.fit(extroversionquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(extroversionquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testextroversionquestions)

    # Evaluation
    openness_perceptron_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions)

    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy))
    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy), file=file_out)

    openness_perceptron_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions)

    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy), file=file_out)

    openness_sgd_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions)

    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy))
    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy), file=file_out)

    openness_sgd_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions)

    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy), file=file_out)

    openness_logistic_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions)

    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy))
    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy), file=file_out)

    openness_logistic_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions)

    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy), file=file_out)

    openness_dt_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions)

    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy))
    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy), file=file_out)

    openness_dt_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions)

    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy), file=file_out)

    return

def PredictBasedOnNeuroticism(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    neuroticismquestions = allsampledataset[:,10:20] # Select neuroticism question columns
    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testneuroticismquestions = alltestdataset[:,10:20] # Select neuroticism question columns
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3.fit(neuroticismquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(neuroticismquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testneuroticismquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(neuroticismquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testneuroticismquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(neuroticismquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testneuroticismquestions)
    
    extroversionclf_decisiontree = DecisionTreeRegressor(max_depth=20)
    extroversionclf_decisiontree.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(neuroticismquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testneuroticismquestions)

    # Evaluation
    extroversion_perceptron_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions)

    print("Able to predict extroversion based on neuroticism training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy), file=file_out)

    extroversion_perceptron_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions)

    print("Able to predict extroversion based on neuroticism testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy), file=file_out)

    extroversion_sgd_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions)

    print("Able to predict extroversion based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy), file=file_out)

    extroversion_sgd_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions)

    print("Able to predict extroversion based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy), file=file_out)

    extroversion_logistic_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions)

    print("Able to predict extroversion based on neuroticism training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy), file=file_out)

    extroversion_logistic_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions)

    print("Able to predict extroversion based on neuroticism testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy), file=file_out)

    extroversion_dt_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions)

    print("Able to predict extroversion based on neuroticism training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy), file=file_out)

    extroversion_dt_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions)

    print("Able to predict extroversion based on neuroticism testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy), file=file_out)

    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(neuroticismquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(neuroticismquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testneuroticismquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(neuroticismquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testneuroticismquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(neuroticismquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testneuroticismquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(neuroticismquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testneuroticismquestions)

    # Evaluation
    agreeableness_perceptron_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions)

    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy), file=file_out)

    agreeableness_perceptron_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy), file=file_out)

    agreeableness_sgd_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions)

    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy), file=file_out)

    agreeableness_sgd_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy), file=file_out)

    agreeableness_logistic_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions)

    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy), file=file_out)

    agreeableness_logistic_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy), file=file_out)

    agreeableness_dt_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions)

    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy), file=file_out)

    agreeableness_dt_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions)

    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy), file=file_out)

    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(neuroticismquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(neuroticismquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testneuroticismquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(neuroticismquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testneuroticismquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(neuroticismquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testneuroticismquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(neuroticismquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testneuroticismquestions)

    # Evaluation
    conscientiousness_perceptron_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions)

    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy), file=file_out)

    conscientiousness_perceptron_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy), file=file_out)

    conscientiousness_sgd_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions)

    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy), file=file_out)

    conscientiousness_sgd_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy), file=file_out)

    conscientiousness_logistic_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions)

    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy), file=file_out)

    conscientiousness_logistic_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy), file=file_out)

    conscientiousness_dt_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions)

    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy), file=file_out)

    conscientiousness_dt_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions)

    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy), file=file_out)

    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3.fit(neuroticismquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(neuroticismquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testneuroticismquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd.fit(neuroticismquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(neuroticismquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testneuroticismquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l2', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(neuroticismquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(neuroticismquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testneuroticismquestions)
    
    opennessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree.fit(neuroticismquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(neuroticismquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testneuroticismquestions)

    # Evaluation
    openness_perceptron_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions)

    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy))
    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy), file=file_out)

    openness_perceptron_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions)

    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy), file=file_out)

    openness_sgd_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions)

    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy))
    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy), file=file_out)

    openness_sgd_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions)

    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy), file=file_out)

    openness_logistic_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions)

    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy))
    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy), file=file_out)

    openness_logistic_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions)

    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy), file=file_out)

    openness_dt_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions)

    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy))
    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy), file=file_out)

    openness_dt_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions)

    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy), file=file_out)

    return

def main():

    #Shrinker() Used for shrinking to predetermined values used while testing
    dataset, trainingdataset, testdataset = ReadInData()
    print("Dataset:\n", dataset)
    print("\n\n\n")
    #print("Trainingdataset:\n", trainingdataset)
    #print("\n\n\n")
    #print("Testingdataset:\n", testdataset)
    #print("\n\n\n")
    datasettotals_training = CalculateIndividualTotals(trainingdataset) # Totals without transforming/normalizing
    datasettotals_testing = CalculateIndividualTotals(testdataset) # Totals without transforming/normalizing
    #print("datasettotals_training:\n", datasettotals_training)
    #print("datasettotals_testing:\n", datasettotals_testing)
    print("\n\n\n")
    datasetpreferences_training = CalculateIndividualPreferences(datasettotals_training) # Preferences (labels) for each trait calculated
    datasetpreferences_testing = CalculateIndividualPreferences(datasettotals_testing) # Preferences (labels) for each trait calculated
    print("datasetpreferences_training:\n", datasetpreferences_training)
    print("datasetpreferences_testing:\n", datasetpreferences_testing)
    print("\n\n\n")
    cleandataset_training = CleanDataPart1(trainingdataset)
    cleandataset_testing = CleanDataPart1(testdataset)
    #print("cleandataset_training:\n", cleandataset_training)
    #print("cleandataset_testing:\n", cleandataset_testing)
    print("\n\n\n")
    cleandatasettotals_training = CalculateCleanedTotals(cleandataset_training) # Totals with transforming/normalizing
    cleandatasettotals_testing = CalculateCleanedTotals(cleandataset_testing) # Totals with transforming/normalizing
    #print("cleandatasettotals_training:\n", cleandatasettotals_training)
    #print("cleandatasettotals_testing:\n", cleandatasettotals_testing)
    print("\n\n\n")
    normalizeddataset_training = NormalizeData(cleandataset_training)
    normalizeddataset_testing = NormalizeData(cleandataset_testing)
    #print("normalizeddataset_training:\n", normalizeddataset_training)
    #print("normalizeddataset_testing:\n", normalizeddataset_testing)
    print("\n\n\n")
    normalizeddatasettotals_training = NormalizeData(cleandatasettotals_training)
    normalizeddatasettotals_testing = NormalizeData(cleandatasettotals_testing)
    #print("normalizeddatasettotals_training:\n", normalizeddatasettotals_training)
    #print("normalizeddatasettotals_testing:\n", normalizeddatasettotals_testing)
    print("\n\n\n")
    cleaneddatasetpreferences_training = CalculateCleanedPreferences(cleandatasettotals_training)
    cleaneddatasetpreferences_testing = CalculateCleanedPreferences(cleandatasettotals_testing)
    print("cleaneddatasetpreferences_training:\n", cleaneddatasetpreferences_training)
    print("cleaneddatasetpreferences_testing:\n", cleaneddatasetpreferences_testing)
    print("\n\n\n")

    print("Done normalizing, calculating totals, and calculating preferences...")

    # prunedindexes, weight, averageweight = PerceptronForPruning(trainingdataset, datasetpreferences_training, (10, 20), [0, 2, 3, 4], 5, 1, 3)
    # print("Weight:", weight)
    # print("Averageweight:", averageweight)
    # print("Prunedindexes:", prunedindexes)
    # print("\n\n\n")

    # print("Done doing perceptron pruning....")

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    debugoutput = open('output.txt', 'w')

    PredictBasedOnExtroversion(debugoutput, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)

    debugoutput.close()
    #file_out = open('output.txt', 'w')
    
    #file_out.close()

    return

main()