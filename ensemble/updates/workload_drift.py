import numpy as np
import math

def chebyshev_single(sample, cov, mu, k):
    ans = (((sample-mu).T).dot(cov.T)).dot(sample-mu)
    if np.linalg.norm(ans)>=k:
        return True
    else:
        return False

def chebyshev_inequality(old_samples, new_samples, k):
    cov = np.cov(old_samples)
    mu = np.mean(old_samples, axis=0)
    N = ((cov.T)*cov).trace()

    sum = 0
    for sample in new_samples:
        if chebyshev_single(sample, cov, mu, k):
            sum += 1
    pr = sum / len(new_samples)
    nk2 = N/(k*k)
    
    return pr, nk2

# test
old_samples = [[1,2,3],[4,5,6]]
new_samples = [[1,3,5],[2,4,6]]
chebyshev_inequality(old_samples, new_samples, 0.05)