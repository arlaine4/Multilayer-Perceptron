import numpy as np

"""Manual if you don't want to use numpy methods to get math values"""

def mean(array, manual=False):
    """Mean of a numpy array"""
    if manual:
        total_size = len(array)
        total = 0.0
        for elem in array:
            total += elem
        total /= total_size
        #print("{:.25f}".format(total[0]))
    return np.mean(array) if not manual else total[0]

def max(array, manual=False):
    """Return maximum value of a numpy array"""
    if manual:
        max_ = array[0]
        for elem in array:
            max_ = elem if elem > max_ else max_
    return np.amax(array) if not manual else max_[0]

def min(array, manual=False):
    """Return minimal value of a numpy array"""
    if manual:
        min_ = array[0]
        for elem in array:
            min_ = elem if elem < min_ else min_
    return np.amin(array) if not manual else min_[0]

def standard_error(array, manual=False):
    """Return standard error (Erreur type / SE) of a numpy array"""
    if manual:
        se_ = 0
        size = len(array)
        se_ = standard_deviation(array, True) / (size**0.5) #ecart type / racine de la taille de l'echantillon
    return np.std(array) / np.size(array)**0.5 if not manual else float(se_)
    #return standard_deviation(array) / (np.size(array)**0.5) if not manual else float(se_)

######


def standard_deviation(array, manual=False):
    """Return standard deviation (Ecart type) of a numpy array"""
    if manual:
        moy = mean(array)
        size = len(array)
        std = 0
        for elem in array:
            std = std + ((elem - moy)**2) #somme de elem - moyenne au carre
        std = (std / size)**0.5
    return np.std(array) if not manual else float(std)
    
def variance(array, manual=False):
    if manual:
        var_ = 0
        moy = mean(array, True)
        size = len(array)
        ecart = 0
        for elem in array:
            ecart = ecart + ((elem - moy)**2)
        var_ = (ecart / size)**0.5
    return np.std(array) / np.mean(array) if not manual else var_
    #return standard_deviation(array) / mean(array) if not manual else var_


########
