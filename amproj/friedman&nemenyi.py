import util
import scipy.stats as stats
import numpy as np
import stac as tst
#import stac.nonparametric_tests.friedman_test

#nemenyi_multitest

viewShapeKnnResult=[41.3,42.0,41.5,42.2,41.9,41.6,42.1,41.4,41.4,42.3,41.9,42.3,41.9,40.8,41.5,41.2,41.8,41.8,41.9,42.7,41.4,41.6,42.1,41.4,42.2,42.2,41.9,41.6,41.4,42.4]

viewRGBKnnResult=[12.3,12.5,12.4,12.7,12.8,12.4,12.7,12.4,12.1,13.5,12.8,12.2,12.8,13.1,12.6,12.9,12.7,12.6,12.9,12.8,12.5,12.9,11.9,12.6,12.7,13.3,12.5,12.5,13.0,13.0]

viewShapeBayesResult=[50.9,51.0,51.1,51.4,51.6,51.2,51.2,51.5,51.4,51.2,51.6,51.5,51.6,51.7,50.9,51.3,50.7,51.3,51.5,51.5,51.4,51.3,51.3,51.7,50.9,51.2,51.2,51.1,51.2,51.3]

viewRGbBayesResult=[19.8,19.8,19.9,20.1,20.1,20.1,19.8,10.2,20.0,19.9,20.3,20.1,20.1,20.2,19.7,20.7,19.9,20.0,19.9,20.0,19.9,19.9,19.9,20.0,20.2,20.1,20.0,19.9,19.9,20.2]

VMJTResult=[17.5,17.7,17.6,17.6,18.6,17.6,18.1,17.4,17.2,18.4,17.9,17.7,17.6,18.0,17.9,17.5,17.4,17.7,17.9,17.5,17.8,17.3,17.4,17.9,18.0,18.2,18.0,17.4,17.8,18.2]

friedman = tst.friedman_test(viewShapeKnnResult,viewRGBKnnResult,viewShapeBayesResult,viewRGbBayesResult,VMJTResult)


'''
print np.mean(viewShapeKnnResult)
print np.mean(viewRGBKnnResult)
print np.mean(viewShapeBayesResult)
print np.mean(viewRGbBayesResult)
print np.mean(VMJTResult)


util.Ic(viewShapeKnnResult)
util.Ic(viewRGBKnnResult)
util.Ic(viewShapeBayesResult)
util.Ic(viewRGbBayesResult)
util.Ic(VMJTResult)

'''


