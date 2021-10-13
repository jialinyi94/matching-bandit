import numpy as np
import copy
class SEBElim(object):
    """
    Implementation of single-entry based elimination process
    """
    def __init__(self, j, T, R_a,R, j_type='column'):
        self.j = j
        self.T = T
        self.candidates = copy.copy(R_a)
        self.j_type = j_type
        self.sum_samples = np.zeros(R)

        self.l = 0
        self.Delta_l = 1
        self.p = 0

    def update(self, X):
        k_l = int(np.ceil(4 / self.Delta_l**2 * np.log(self.T)))
        self.p += 1
        self.sum_samples += X

        D_R = []
        i_l = -1
        if self.p == k_l:
            exploration = np.sqrt(np.log(self.T) / k_l)
            U = self.sum_samples / k_l + exploration
            L = self.sum_samples / k_l - exploration
            
            # Elimination
            i_l = self.candidates[0]
            for i in self.candidates:
                if L[i] > L[i_l]:
                    i_l = i

            for i in self.candidates:
                if U[i] < L[i_l]:
                    D_R.append(i)
            for i in D_R:
                self.candidates.remove(i)

            self.l += 1
            self.Delta_l /= 2
        
        return D_R, i_l
        
    def remove(self,D_R):
        for i in D_R:
            if i in self.candidates:
                self.candidates.remove(i)