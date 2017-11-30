
import numpy as np

def compute_n_batches(n_samples, batch_size):
    n_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches

def compute_batch_idxs(start, batch_size, size, fill='random'):
    if start >= size:
        return list(np.random.randint(low=0, high=size, size=batch_size))
    
    end = start + batch_size

    if end <= size:
        return list(range(start, end))

    else:
        base_idxs = list(range(start, size))
        if fill == 'none':
            return base_idxs
        elif fill == 'random':
            remainder = end - size
            idxs = list(np.random.randint(low=0, high=size, size=remainder))
            return base_idxs + idxs
        else:
            raise ValueError('invalid fill: {}'.format(fill))
       
def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return lengths