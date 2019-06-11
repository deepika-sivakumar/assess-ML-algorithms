
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import time

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

def generate_graph(filename, title, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
#    plt.show()
    plt.savefig(filename)
    plt.close()

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    testX,testY,trainX,trainY = None,None, None,None
    # Print only if verbose is set to true
    verbose = False

    data = np.genfromtxt(inf,delimiter=',')
#     Skip the date column and header row if we're working on Istanbul data
    if sys.argv[1].find('Istanbul.csv') != -1:
        data = data[1:,1:]

#    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows
    if(verbose == True):
        print('testLearner::train_rows:',train_rows)
        print('testLearner::test_rows:',test_rows)
    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]
#    print('testLearner::trainX:',trainX)

    if(verbose == True):
        print('testLearner::trainX:',trainX)
        print('testLearner::trainY:',trainY)
        print('testLearner::testX:',testX)
        print('testLearner::testY:',testY)
        print testX.shape
        print testY.shape

    # create a learner and train it
    learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    c = np.corrcoef(predY, y=trainY)
    if(verbose == True):
        print
        print "In sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c = np.corrcoef(predY, y=testY)
    if(verbose == True):
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # Create a DTLearner and train it
    dtlearner = dt.DTLearner(leaf_size = 1, verbose = False) # constructor
    dtlearner.addEvidence(trainX, trainY) # train it
    if(verbose == True):
        print dtlearner.author()

    #evaluate in sample
    predY = dtlearner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    c = np.corrcoef(predY, y=trainY)
    if(verbose == True):
        print
        print "DTLearner In sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # evaluate out of sample
    predY = dtlearner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c = np.corrcoef(predY, y=testY)
    if(verbose == True):
        print
        print "DTLearner Out of sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # Create a RTLearner and train it
    np.random.seed(1481090002)
    rtlearner = rt.RTLearner(leaf_size = 1, verbose = False) # constructor
    rtlearner.addEvidence(trainX, trainY) # training step
    if(verbose == True):
        print rtlearner.author()

    #evaluate in sample for the RTLearner
    predY = rtlearner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    c = np.corrcoef(predY, y=trainY)
    if(verbose == True):
        print
        print "RTLearner In sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # evaluate out of sample
    predY = rtlearner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c = np.corrcoef(predY, y=testY)
    if(verbose == True):
        print
        print "RTLearner Out of sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # Create a BagLearner with DTLearner Ensemble and train it
    np.random.seed(1481090001)
    baglearner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":1, "verbose":False}, bags = 10, boost = False, verbose = False)
    baglearner.addEvidence(trainX, trainY)
    if(verbose == True):
        print baglearner.author()

    # evaluate in sample for the BagLearner
    predY = baglearner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    c = np.corrcoef(predY, y=trainY)
    if(verbose == True):
        print
        print "BagLearner with DTLearner In sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # evaluate out of sample
    predY = baglearner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c = np.corrcoef(predY, y=testY)
    if(verbose == True):
        print
        print "BagLearner with DTLearner Out of sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # Create a BagLearner with RTLearner Ensemble and train it
    np.random.seed(1481090002)
    baglearner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1, "verbose":False}, bags = 10, boost = False, verbose = False)
    baglearner.addEvidence(trainX, trainY)
    if(verbose == True):
        print baglearner.author()

    # evaluate in sample for the BagLearner
    predY = baglearner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    c = np.corrcoef(predY, y=trainY)
    if(verbose == True):
        print
        print "BagLearner with RTLearner In sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # evaluate out of sample
    predY = baglearner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c = np.corrcoef(predY, y=testY)
    if(verbose == True):
        print
        print "BagLearner with RTLearner Out of sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # Create a InsaneLearner and train it
    np.random.seed(1498076428)
    insanelearner = it.InsaneLearner(verbose = False) # constructor
    insanelearner.addEvidence(trainX, trainY) # training step
    if(verbose == True):
        print insanelearner.author()

    # evaluate in sample for the InsaneLearner
    predY = insanelearner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    c = np.corrcoef(predY, y=trainY)
    if(verbose == True):
        print
        print "InsaneLearner In sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # evaluate out of sample for the InsaneLearner
    predY = insanelearner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c = np.corrcoef(predY, y=testY)
    if(verbose == True):
        print
        print "InsaneLearner Out of sample results"
        print "RMSE: ", rmse
        print "corr: ", c[0,1]

    # Report plots
    # Overfitting wrt leaf size in DTLearner? Metric: RMSE
    # Create a DTLearner and train it
    leaf_size_list = list(range(1,51))
    in_sample_rmse = []
    out_sample_rmse = []
    # Train & query the DTLearner in sample and out sample for leaf sizes 1 through 50
    for i in range(50):
        dtlearner = dt.DTLearner(leaf_size = leaf_size_list[i], verbose = False) # constructor
        dtlearner.addEvidence(trainX, trainY) # train it
        # Query in sample and store the rmse values
        predInSampleY = dtlearner.query(trainX) 
        in_rmse = math.sqrt(((trainY - predInSampleY) ** 2).sum()/trainY.shape[0])
        in_sample_rmse.append(in_rmse)
        # Query out sample and store the rmse values
        predOutSampleY = dtlearner.query(testX) # get the predictions
        out_rmse = math.sqrt(((testY - predOutSampleY) ** 2).sum()/testY.shape[0])
        out_sample_rmse.append(out_rmse)
    # Plot the graph
    plt.plot(leaf_size_list, in_sample_rmse, linewidth=2, label='In Sample')
    plt.plot(leaf_size_list, out_sample_rmse, linewidth=1, label='Out sample')
    generate_graph('graph_1','Does overfitting occur wrt leaf_size in DTLearner? \n Data: Istanbul.csv', 'Leaf Size', 'RMSE')

    # Bagging reduces/eliminates overfitting wrt leaf size? Metric: RMSE
    # Create a BagLearner and train it
    leaf_size_list = list(range(1,51))
    in_sample_rmse = []
    bag_out_sample_rmse = []
    np.random.seed(1481090004)
    # Train & query the BagLearner in sample and out sample for fixed bags=20, leaf sizes 1 through 50
    for i in range(50):
        baglearner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":leaf_size_list[i], "verbose":False}, bags = 20, boost = False, verbose = False)
        baglearner.addEvidence(trainX, trainY) # train it
        # Query in sample and store the rmse values
        predInSampleY = baglearner.query(trainX) 
        in_rmse = math.sqrt(((trainY - predInSampleY) ** 2).sum()/trainY.shape[0])
        in_sample_rmse.append(in_rmse)
        # Query out sample and store the rmse values
        predOutSampleY = baglearner.query(testX) # get the predictions
        out_rmse = math.sqrt(((testY - predOutSampleY) ** 2).sum()/testY.shape[0])
        bag_out_sample_rmse.append(out_rmse)

    # Plot the graph
    plt.plot(leaf_size_list, in_sample_rmse, linewidth=2, label='In Sample')
    plt.plot(leaf_size_list, bag_out_sample_rmse, linewidth=1, label='Out sample')
    generate_graph('graph_2','Does bagging reduce/eliminate overfitting wrt leaf_size in DTLearner? \n Data: Istanbul.csv', 'Leaf Size', 'RMSE')

    # Out of sample RMSE graph DTLearner with & without bagging
    plt.plot(leaf_size_list, out_sample_rmse, linewidth=2, label='Without bagging')
    plt.plot(leaf_size_list, bag_out_sample_rmse, linewidth=1, label='With bagging')
    generate_graph('graph_3','DTLearner Out sample RMSE with & without bagging \n Data: Istanbul.csv', 'Leaf Size', 'RMSE')

    # Comparing DTLearner Vs RTLearner
    # Performance metrics: Accuracy with Mean Absolute Error (mae) & Time
    leaf_size_list = list(range(1,51))
    dt_mae_list = []
    rt_mae_list = []
    dt_time_list = []
    rt_time_list = []
    np.random.seed(1481090003)
    # Train & query out of sample for leaf sizes 1 through 50
    for i in range(50):
        # DTLearner
        dtlearner = dt.DTLearner(leaf_size = leaf_size_list[i], verbose = False) # constructor
        dt_start_time = time.time()
        dtlearner.addEvidence(trainX, trainY) # train it
        dt_end_time = time.time()
        # Calculate total DTLearner tree building time as end time - start time
        dt_time_list.append(dt_end_time - dt_start_time)
        # Query out sample and store the mae values
        predDtSampleY = dtlearner.query(testX) # get the predictions
        dt_mae = abs(testY - predDtSampleY).sum()/testY.shape[0]
        dt_mae_list.append(dt_mae)
        # RTLearner
        rtlearner = rt.RTLearner(leaf_size = leaf_size_list[i], verbose = False) # constructor
        rt_start_time = time.time()
        rtlearner.addEvidence(trainX, trainY) # train it
        rt_end_time = time.time()
        # Calculate total DTLearner tree building time as end time - start time
        rt_time_list.append(rt_end_time - rt_start_time)
        # Query out sample and store the mae values
        predRtSampleY = rtlearner.query(testX) # get the predictions
        rt_mae = abs(testY - predRtSampleY).sum()/testY.shape[0]
        rt_mae_list.append(rt_mae)

    # Plot the graph for MAE
    plt.plot(leaf_size_list, dt_mae_list,color='gold', linewidth=2, label='DT Learner')
    plt.plot(leaf_size_list, rt_mae_list,color='purple', linewidth=2, label='RT Learner')
    generate_graph('graph_4','Comparing DTLearner Vs RTLearner \n Metric: Mean Absolute Error \n Data: Istanbul.csv', 'Leaf Size', 'MAE')

    #Plot the graph for Time
    plt.plot(leaf_size_list, dt_time_list,color='gold', linewidth=2, label='DT Learner')
    plt.plot(leaf_size_list, rt_time_list,color='purple', linewidth=2, label='RT Learner')
    generate_graph('graph_5','Comparing DTLearner Vs RTLearner \n Metric: Tree building time \n Data: Istanbul.csv', 'Leaf Size', 'Time')



