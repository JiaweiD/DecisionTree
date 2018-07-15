import operator
from math import log

#calculate Shannon Entrophy of a dataset
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#split a dataset by a certain value of a certain axis
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#choose the best feature to split by information gain
def chooseBestFeatureToSplitByGain(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntrophy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntrophy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntrophy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntrophy - newEntrophy
        if  (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#choose the best feature to split by information gain(when dealing with continuous attributes) 
def chooseBestContinuousFeatureToSplitByGain(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntrophy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestAttrInfoGain = 0.0
    bestFeature = -1
    bestAttrValue = 0
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        midPointList = [(float(featList[i])+float(featList[i+1]))/2 for i in range(len(featList)-1)]
        newEntrophy = 0.0
        for value in midPointList:
            subDataSetLess = []
            subDataSetMore = []
            for dataVec in dataSet:
                if float(dataVec[i]) < value:
                    reducedVec = dataSet[:i]
                    reducedVec.extend(dataSet[i+1:])
                    subDataSetLess.extend(reducedVec)
                else:
                    reducedVec = dataSet[:i]
                    reducedVec.extend(dataSet[i + 1:])
                    subDataSetMore.extend(reducedVec)
            probLess = len(subDataSetLess)/float(len(dataSet))
            probMore = len(subDataSetMore)/float(len(dataSet))
            attrEntrophy = probLess * calcShannonEnt(subDataSetLess) + probMore * calcShannonEnt(subDataSetMore)
            attrInfoGain = baseEntrophy - attrEntrophy
            if attrInfoGain > bestAttrInfoGain:
                bestAttrInfoGain = attrInfoGain
                bestAttrValue = value
                newEntrophy = attrInfoGain
        infoGain = baseEntrophy - newEntrophy
        if  (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#choose the best feature to split by information gain ratio(C4.5)
def chooseBestFeatureToSplitByGainRatio(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntrophy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntrophy = 0.0
        valuesOfAttributeI = {}
        IntrinsicValue = 0
        for value in featList:
            if value not in valuesOfAttributeI:
                valuesOfAttributeI.setdefault(value,0)
            valuesOfAttributeI[value]+=1
        for key in valuesOfAttributeI:
            IntrinsicValue-=valuesOfAttributeI[key]/len(featList)*log(valuesOfAttributeI[key]/len(featList),2)
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntrophy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntrophy - newEntrophy
        infoGainRatio = infoGain/IntrinsicValue
        if  (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature

#calculate the gini index of a dataset
def calcGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        gini -= prob**2
    return gini

#choose the best feature to split by gini index(CART)
def chooseBestFeatureToSplitByGini(dataSet):
    numFeatures = len(dataSet[0])-1
    bestGiniIndex = 99999; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newGiniIndex = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newGiniIndex += prob * calcGini(subDataSet)
        if  (newGiniIndex < bestGiniIndex):
            bestGiniIndex = newGiniIndex
            bestFeature = i
    return bestFeature

#deciding the classification of the leaf nodes by voting
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#create a decision tree
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitByGain(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#classifying function when testing
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:   classLabel = secondDict[key]
    return classLabel

#store trees
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    print(pickle.load(fr))
