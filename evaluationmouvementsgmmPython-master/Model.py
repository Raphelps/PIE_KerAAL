
class Model:
    def __init__(self, nbVar, nbVarMan, nbStates, dt, params_diagRegFact):
        self.x = None
        self.cuts = None
        self.cutsKP = None
        self.Priors = None
        self.Mu = None
        self.Sigma = None
        self.MuMan = None
        self.U0 = None
        self.nbVar = nbVar                              #Dimension of the tangent space (incl. time)
        self.nbVarMan = nbVarMan                        #Dimension of the manifold (incl. time)
        self.nbStates = nbStates                        #Number of states in the GMM
        self.dt = dt                                    #Time step duration
        self.params_diagRegFact = params_diagRegFact    #Regularization of covariance

