import sys
import scipy.io.arff as sci
import numpy as np
import random
import math
import pylab

def compEntropy(y):
    countLabels = {i:0 for i in labels}

    totalN = len(y)
    for label in y:
        countLabels[label] += 1   
    entropy = 0
    for i in countLabels:
        if countLabels[i] != 0 :
            entropy += -countLabels[i]/float(totalN) * np.log2(countLabels[i]/float(totalN))
    return entropy, countLabels
    
def compSplit(xi, y):
    pos = -1
    minH = 1
    totalN = len(y)
    for i in xrange(1, totalN):
        if xi[i] != xi[i-1]:
            currH = i/float(totalN) * compEntropy(y[:i])[0] + (totalN - i)/float(totalN) * compEntropy(y[i:])[0]
            if currH < minH:
                minH = currH
                pos = i
    return pos, minH

def createEmptyLeafNode():
    L = {i:0 for i in labels}
    return ('Leaf', labels[0], L)
     
def createTree(trSet, m):
    if trSet.shape[0] == 0:
        return None

    else:
        H, labelCnt = compEntropy(trSet[:,-1])
        majorityCnt = 0;
        majorityLabel = labels[0]

        for i in labels:
            if labelCnt[i] > majorityCnt:
                majorityCnt = labelCnt[i]
                majorityLabel = i

        noFeatureSplit = True;
        startLabel = trSet[:,:-1][0];
        for i in xrange(1, trSet[:,:-1].shape[0]):
            tmp = trSet[:,:-1][i] == startLabel
            if (min(tmp) != True) :
                noFeatureSplit = False;
                break;

        if trSet.shape[0] < m or max(trSet[:,-1]) == min(trSet[:,-1]) or noFeatureSplit :
            return ('Leaf', majorityLabel, labelCnt);
    
        else:
            selectFeature = -1
            maxInfoGain = -1

            for i in xrange(len(features) - 1): 
                if classNames[i][0] == 'nominal':
                    tmpSet = trSet[trSet[:,i].argsort()] 
                    
                    f = tmpSet[:,i]
                    l = tmpSet[:,-1]

                    cnt = [sum(f == j) for j in classNames[i][1]]
                    E = [compEntropy(l[f == j])[0] for j in classNames[i][1]]
                    sumE = sum([(cnt[index] * E[index]) for index in xrange(len(cnt))])
                     
                    H_norminali = sumE / float(len(l))

                    if H - H_norminali > maxInfoGain:
                        maxInfoGain = H - H_norminali
                        selectFeature = i

                elif classNames[i][0] == 'numeric':
                    tmpSet = trSet[np.array(trSet[:,i], dtype = 'float64').argsort()] 
                    pos, H_afterSplit = compSplit(tmpSet[:,i], tmpSet[:,-1])

                    if H - H_afterSplit > maxInfoGain:
                        maxInfoGain = H - H_afterSplit
                        selectFeature = i
                        threshold = 0.5 * (float(tmpSet[:,i][pos]) + float(tmpSet[:,i][pos-1]))
                    
            if maxInfoGain <= 0:
                return ('Leaf', majorityLabel, labelCnt);          
        
            if classNames[selectFeature][0] == 'nominal':
                children = []
                for n in classNames[selectFeature][1]:
                    nextLevelNode = createTree(trSet[trSet[:,selectFeature] == n, :], m)
                    if nextLevelNode == None:
                        nextLevelNode = createEmptyLeafNode() 
                    children.append(nextLevelNode)
                return ((selectFeature, None, labelCnt, False), tuple(children))   

            elif classNames[selectFeature][0] == 'numeric':
                temp = np.array(trSet[:,selectFeature], dtype = 'float64')
                left = createTree(trSet[temp <= threshold, :], m)
                right = createTree(trSet[temp > threshold, :], m)

                if left == None:
                    left = createEmptyLeafNode() 
                if right == None:
                    right = createEmptyLeafNode()
                return ((selectFeature, threshold, labelCnt, True), (left, right))
                       

def testTree(trainSet, root):
    print "--------------------------------------------------------------------------------------"
    print " Testing Results:"
    print ('{0:15s}|{1:16s}|{2:16s}'.format("Test index", "Accurate label", "Predicted label"))

    correctCnt = 0
    for i in xrange(trainSet.shape[0]): 

        currNode = root
        
        while (len(currNode) == 2):
            featureI = currNode[0][0]

            if (currNode[0][3] == False):
                names = classNames[featureI][1]
                for j in xrange(len(names)):
                    if (names[j] == trainSet[i][featureI]):
                        classId = j;
                        break;
                currNode = currNode[1][classId]
            else:
                if float(trainSet[i][featureI]) <= currNode[0][1]:
                    currNode = currNode[1][0]
                else:
                    currNode = currNode[1][1]

        predict = currNode[1]
        print('{0:<15d}|{1:16s}|{2:16s}'.format(i+1, trainSet[i,-1], predict))

        if predict == trainSet[i,-1]:
            correctCnt += 1

    print "correct classified instances: ", correctCnt
    print "total classified instances: ", trainSet.shape[0]  

    return correctCnt, trainSet.shape[0]  
        
def printLeaf(currNode):
    if currNode[2] != None :      
        print ' ', currNode[2], ': ' + currNode[1]
    else :                
        print  ' : ' + currNode[1]

def visualizeTree(root, level):
    if (len(root) != 2):
        return

    indent = level * 8 * (' ')

    if root[0][3] == False: 
        featureId = root[0][0]
        for i in range(len(classNames[featureId][1])): 
            nodeInfo = root[1][i]
            print indent + '(' + features[featureId] + ':' + classNames[featureId][1][i] + ')' + ':',   
            if len(nodeInfo) == 3:
                printLeaf(nodeInfo)
            else:
                print ' ', nodeInfo[0][2]
                visualizeTree(nodeInfo, level+1)
    else: 
        nodeInfo = root[1][0]
        print indent + '(' + features[root[0][0]] + '<=' + str(root[0][1]) + ')' + ':',    
        if len(nodeInfo) == 3:  
            printLeaf(root[1][0])
        else:
            print ' ', nodeInfo[0][2]
            visualizeTree(nodeInfo, level+1)

        print indent + '(' + features[root[0][0]] + '>' + str(root[0][1]) + ')' + ':',
        nodeInfo = root[1][1]
        if len(nodeInfo) == 3:       
            printLeaf(nodeInfo)
        else:
            print ' ', nodeInfo[0][2]       
            visualizeTree(nodeInfo, level+1)       
                                                                                                                                                          
args = [arg for arg in sys.argv]
try:
    trainFile = args[1]
    testFile = args[2] 
    m = int(args[3])

    trainData = sci.loadarff(trainFile)
    testData = sci.loadarff(testFile)  
   
    train = np.array([ [i for i in trainData[0][j]] for j in range(trainData[0].shape[0])])
    test = np.array([ [i for i in testData[0][j]] for j in range(testData[0].shape[0])])
    
    classNames = [trainData[1][i] for i in trainData[1].names()]
 
    features = trainData[1].names()
    labels = classNames[-1][1]
    root = createTree(train, m)
    
    ## visualize the tree structure
    print "--------------------------------------------------------------------------------"
    print " Visualize the decision tree:"
    visualizeTree(root, 0)
    
    testTree(test, root)

except:
      print 'python dt-learn.py <train-set-file> <test-set-file> m'
    

