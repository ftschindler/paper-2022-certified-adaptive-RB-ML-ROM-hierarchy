import numpy as np

def compute_expectation_and_variance(outputs):
    array_estimated_expectation = np.cumsum(np.array(outputs))
    array_estimated_expectation = array_estimated_expectation / np.arange(1, len(array_estimated_expectation) + 1)
    array_estimated_variance = np.cumsum((np.array(outputs) - array_estimated_expectation)**2)
    array_estimated_variance = array_estimated_variance / np.maximum(np.arange(len(array_estimated_expectation)), 1)

    return array_estimated_expectation, array_estimated_variance
