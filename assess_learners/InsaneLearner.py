import BagLearner as bl
import LinRegLearner as lrl


class InsaneLearner:
    N_LEARNERS, BAGS_PER_LEARNER = 20, 20

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._model = None

    def author(self):
        return 'cfarr31'  # replace tb34 with your Georgia Tech username.

    def addEvidence(self, Xtrain, Ytrain):
        self._model = bl.BagLearner(learner=bl.BagLearner,
                                    kwargs={"bags": InsaneLearner.BAGS_PER_LEARNER, "learner": lrl.LinRegLearner,
                                            "kwargs": {}, "boost": True}, bags=InsaneLearner.N_LEARNERS,
                                    boost=False).addEvidence(Xtrain, Ytrain)
        return self

    def query(self, Xtest):
        return self._model.query(Xtest)
