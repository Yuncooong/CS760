import sys
import scipy.io.arff as sia
import numpy as np
import random
import math
import pylab
import re

def trainNB(classes, train):
    '''
    estimate the prior probabilities P(Yj) and conditional probabilities P(Xi|Yj)
    '''
    (N,D) = train.shape
    Xs = classes[:-1]
    Ys = classes[-1]  
    nY = len(Ys)
    subtrain = [train[train[:,-1]==l,:] for l in Ys]

    positiveTrains = []
    negativeTrains = []

    for i in range(N):
        curr = train[i]
        if curr[-1] == Ys[0]:
            positiveTrains.append(curr)
        else :
            negativeTrains.append(curr)

    Py1 = float(len(positiveTrains)+1) / (N+nY)
    Py2 = float(len(negativeTrains)) / (len(train))


    Pfeature = []
    for feature_index in xrange(len(Xs)):
        f = []
        
        for feature_class in Xs[feature_index]:
            p = []
            posCnt = 0
            negCnt = 0
            for instance in positiveTrains:
                if instance[feature_index] == feature_class:
                    posCnt = posCnt+1;
        
            p.append((posCnt+1) / float(len(positiveTrains) + len(Xs[feature_index])))
            for instance in negativeTrains:
                if instance[feature_index] == feature_class:
                    negCnt = negCnt+1;
            
            p.append((negCnt+1)/float(len(negativeTrains)+len(Xs[feature_index])))
            f.append(p)
        Pfeature.append(f)
    return[Pfeature, [Py1, Py2]]

def testNB(P, test, classes, display=True):
    '''
    estimate postieria probability P(Yi|X)
    '''
    Pxy = P[0]
    Py = P[1]
    [Nt, D] = test.shape
    nY = len(classes[-1])
    logres = np.ones((Nt,nY))
    corr = 0
    for i in xrange(test.shape[0]):
        for j in xrange(len(classes[-1])):
            curr = test[i,:]
            Psum = 0
            for feature_index in xrange(test.shape[1]-1):
                classIndex = classes[feature_index].index(curr[feature_index])
                Psum += math.log(Pxy[feature_index][classIndex][j])
            logres[i,j] = Psum + math.log(Py[j])
            if (j == len(classes[-1])-1) :
                if (logres[i,0] > logres[i,1]) :
                    label = 0
                else :
                    label = 1
                if (classes[-1][label] == test[i][-1]):
                    corr += 1
    
    res = np.exp(logres)
    res = res/np.sum(res,1).reshape(-1,1)
    
    ## output the prediction result
    if display:
        label_id = res.argmax(1)
        for i in xrange(len(test)):
            print classes[-1][label_id[i]],' ', test[i][-1], ' ', res[i,label_id[i]]
        print
        #print corr
        return res, corr
    
    else:
        #print corr  
        return res, corr
    
def correlationCal(classes, train):
    Xs = classes[:-1]
    Ys = classes[-1]
    nY = len(Ys)
    subtrain = [train[train[:,-1]==l,:] for l in Ys]
    
    nodes = len(classes)-1
    correlation = np.random.rand(nodes, nodes)
    [Pxy, Py] = trainNB(classes, train)
    
    for i in range(nodes):
        correlation[i][i] = -1.0
        for j in range(0, i):
            corr_c = []
            corr_p = []

            for kk in range(nY):
                i_corr = []
                ip_corr = []
                for dd in Xs[j] :
                    j_corr = []
                    jp_corr = []
                    for cc in Xs[i] :
                        cnt = 0
                        cnt_p = 0
                        for instance in subtrain[kk]:
                            
                            if (instance[i]==cc and instance[j]==dd):
                                cnt +=1
                        j_corr.append((1+ cnt) / float(subtrain[kk].shape[0] + len(Xs[i])*len(Xs[j])) )  
                        jp_corr.append((1+ cnt) / float(train.shape[0] + len(Xs[i])*len(Xs[j])*nY))
                    i_corr.append(j_corr)
                    ip_corr.append(jp_corr)
                corr_c.append(i_corr)
                corr_p.append(ip_corr)

            tmp = 0
            for ii in range(len(Xs[i])):
                for jj in range(len(Xs[j])):
                    for k in range(nY):
                        tmp += corr_p[k][jj][ii]*math.log(corr_c[k][jj][ii] / (Pxy[i][ii][k]*Pxy[j][jj][k])) / math.log(2)        
            correlation[i][j] = correlation[j][i] = tmp
    return correlation
                
def createSpanningTree(corr):
    N = corr.shape[0]
    treeMap = {0:[N]}
    nodeRemain = range(1,N)
    while len(treeMap) < N:
        currMax = -1
        currEdge = []
        for i in treeMap:
            for j in nodeRemain:
                if corr[i][j] > currMax:
                    currMax = corr[i][j]
                    currEdge = [i,j]
                elif corr[i][j] == currMax:
                    if i<currEdge[0] or (i==currEdge[0] and j<currEdge[1]):
                        currMax = corr[i][j]
                        currEdge = [i,j]
 
        treeMap[currEdge[1]] = [currEdge[0]]+[N]
        nodeRemain.remove(currEdge[1])     
    return treeMap        
 
def compProb(classes, train, x, coor):
    '''
    compute P(x | X)
    '''
    count = dict() 

    if (len(coor) > 0):
        c = coor[0]  
        if train.shape[0] == 0:  
            for i in range(len(classes[c])): 
                count[i] = compProb(classes, train, x, coor[1:]) 
            res = [c, count]
        else: 
            
            for i in range(len(classes[c])): 
                subTrain = train[train[:,c]==classes[c][i],:]      
                count[i] = compProb(classes, subTrain, x, coor[1:]) 
            res = [c, count]
    else:
        if train.shape[0] == 0:
            for i in range(len(classes[x])):
                count[i] = float(1)/len(classes[x])
            res = [x, count]
        else :
            for i in range(len(classes[x])):
                subTrain = train[train[:,x]==classes[x][i],:]   
                count[i] = float(1+subTrain.shape[0])/(train.shape[0]+len(classes[x]))
            res = [x, count]
    return res
            
    
def testTAN(spanningTree, test, classes, display=True):
    [Nt, D] = test.shape
    nY = len(classes[-1])
    logres = np.ones((Nt,nY))
    nCorrect = 0
    for i in xrange(Nt):          
        for j in xrange(nY):
            t = list(test[i])
            t[-1] = classes[-1][j]
            temp = 0
            for k in xrange(D):
                curr_Class = spanningTree[k][0]
                CPT_next = spanningTree[k]
                while curr_Class != k:
                    classIndex = classes[curr_Class].index(t[curr_Class])
                    CPT_next = CPT_next[-1][classIndex]
                    curr_Class = CPT_next[0]
                   
                classIndex = classes[curr_Class].index(t[curr_Class])
                temp += math.log(CPT_next[-1][classIndex])

            logres[i,j] = temp
            if j == len(classes[-1])-1:
                label_id = -1
                if logres[i,0] >= logres[i,1]:
                    label_id = 0
                else:
                    label_id = 1
                if (classes[-1][label_id] == test[i][-1]):
                    nCorrect += 1
            
    res = np.exp(logres)
    res = res/np.sum(res,1).reshape(-1,1) 
    
    if display:
        label_id = res.argmax(1)
        for i in xrange(len(test)):
            print classes[-1][label_id[i]],' ', test[i][-1], ' ', res[i,label_id[i]]
        print
        return res, nCorrect
    else:
        return res, nCorrect
    
def errMsg():
    print 'Usage: python bayes.py <train-set-file> <test-set-file> <n|t>'
    
args = [arg for arg in sys.argv]



try:

    trainFile = args[1]
    testFile = args[2] 
    
    name = re.sub("[^A-Za-z']+", ' ', trainFile)
    name = name[:name.find(' ')]
    
    m = args[3]   # 'n' stands for 'naive bayes', 't' stands for 'TAN'

       
    ## load training and test data
    trainData = sia.loadarff(trainFile)
    testData = sia.loadarff(testFile)  

    '''
    Data = sia.loadarff('chess-KingRookVKingPawn.arff')
    t = np.array([[i for i in Data[0][j]] for j in range(Data[0].shape[0])])
    feats = Data[1].names()
    temp = [Data[1][feat] for feat in Data[1].names()]
    classes = [ line[-1] for line in temp]
    labels = classes[-1]
    tmp = np.array_split(t, 10)
    #print tmp[0]
    print len(t)
    for i in range(10):

        test = tmp[i]
        train = tmp[0]
        index = 0
        for j in range(10):
            if i == 0:
                train = tmp[1]
                index = 1
            if j != i and j != index:
                train = np.concatenate((train, tmp[j]))

            #print train
          
        Prob = trainNB(classes, train)
        (Pyx, nCorrect) = testNB(Prob, test, classes, False)
        correlation = correlationCal(classes, train)
        
        spanningTree = createSpanningTree(correlation)
        spanningTree[len(spanningTree)] = [] 
        condProb = dict()
        for i in spanningTree:
            condProb[i] = compProb(classes, train, i, spanningTree[i])

        (Pyx, nCorrect) = testTAN(condProb, test, classes,False)
        print "testSize", len(test)
        print "Correct Number = %d" %(nCorrect)
                
        #print "nCorrect = %d" %(nCorrect)
        '''


    
    ## reshape the datasets
    train = np.array([[i for i in trainData[0][j]] for j in range(trainData[0].shape[0])])
    test = np.array([[i for i in testData[0][j]] for j in range(testData[0].shape[0])])


    ## get the feature names and the class names
    feats = trainData[1].names()
    temp = [trainData[1][feat] for feat in trainData[1].names()]
    classes = [ line[-1] for line in temp]
    labels = classes[-1]
    
    if m == 'n':

        '''
        Naive Bayes
        '''
        ## estimate the prior probabilities P(Y) and conditional probabilities P(Xi|Y)
        Prob = trainNB(classes, train)
        
        ## output the structure of Naive Bayes Net
        for i in xrange(len(feats)-1):
            print feats[i], ' ', feats[-1]
        print

       
        (Pyx, nCorrect) = testNB(Prob, test, classes)
                
        print "nCorrect = %d" %(nCorrect)
        #temp += nCorrect

        
    elif m == 't': 
           
        # (Xi, Xj | Y) 

        correlation = correlationCal(classes, train)
        
        spanningTree = createSpanningTree(correlation)
        spanningTree[len(spanningTree)] = [] 

                ## estimate the conditional probability table
        condProb = dict()
        for i in spanningTree:
            condProb[i] = compProb(classes, train, i, spanningTree[i])

                ## output the structure of Naive Bayes Net
        for i in xrange(len(feats)-1):
            parent = [feats[j] for j in spanningTree[i]]
            print feats[i], ' '.join(parent)
        print
                    
                ## test with Bayes Rule
        (Pyx, nCorrect) = testTAN(condProb, test, classes)
        print "Correct Number = %d" %(nCorrect)
                #temp += nCorrect
         
    else:
        errMsg()
except:
    errMsg() 
       
