import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    
    # Constructor
    def __init__(self, verbose):
        self.bagLearners = []
        for i in range(20):
            self.bagLearners.append(bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {"verbose":False}, bags = 20, boost = False, verbose = False))
    
    def author(self):
        return 'Deepika'
    
    # Add Training data
    def addEvidence(self,dataX,dataY):
        for i in range(20):
            self.bagLearners[i].addEvidence(dataX, dataY)
        return
    
    # Make Predictions
    def query(self,testX):
        predictY = np.empty((0, int(testX.shape[0])), float)
        for i in range(20):
            predictY = np.vstack((predictY, self.bagLearners[i].query(testX)))
        return np.mean(predictY, axis=0)
