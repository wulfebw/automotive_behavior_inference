
import csv
import h5py
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
    return np.array(lengths)

def write_metadata(filepath, data, labels):
    with open(filepath, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(labels)
        writer.writerows(data)

def normalize(x, lengths):
    # bit complicated due to variables lengths
    # compute mean by summing over length and dividing by length
    # then mean over samples
    mean = np.mean(np.sum(x, 1) / lengths.reshape(-1,1), 0)
    x = x - mean

    # at this point the mean of each feature is 0
    # so the variance is just E[X^2], std = sqrt of that
    std = np.sqrt(np.mean(np.sum(x ** 2, 1) / lengths.reshape(-1,1), 0)) + 1e-8
    x = x / std

    # set everything after the length of the value back to zero
    for i, l in enumerate(lengths):
        x[i, l:] = 0
    return dict(x=x, mean=mean, std=std)

def apply_normalize(x, mean, std, lengths):
    x = (x - mean) / std
    for (i, l) in enumerate(lengths):
        x[i, l:] = 0
    return x

def prepend_timeseries_zero(x):
    return np.concatenate((np.zeros((x.shape[0], 1, x.shape[2])), x), axis=1)

def load_x_feature_names(filepath, mode='artificial'):
    f = h5py.File(filepath, 'r')

    if mode == 'artificial':
        x = f['risk/features']
        feature_names = f['risk'].attrs['feature_names']
    elif mode == 'ngsim':
        x = np.concatenate([f['{}'.format(i)] for i in range(1,6+1)])
        feature_names = f.attrs['feature_names']

    return x, feature_names

def load_metadata(filepath, mode='ngsim', obs_keys=None):
    f = h5py.File(filepath, 'r')
    metadata, meta_labels = None, None

    if mode == 'ngsim':
        traj = []
        for i in range(1, 6+1):
            traj.extend([i] * len(f['{}'.format(i)]))

        feature_names = f.attrs['feature_names']
        idxs = [i for (i,n) in enumerate(feature_names) if n in obs_keys]
        obs_keys = feature_names[idxs]

        x = np.concatenate([f['{}'.format(i)] for i in range(1,6+1)])
        x = x[:,:,idxs]
        lengths = compute_lengths(x)
        feature_means = []
        
        for i, l in enumerate(lengths):
            sample_features = x[i,:l]
            feature_means.append(x[i,:l].mean(0))

        n_meta = len(obs_keys) + 2
        metadata = np.zeros((len(traj), n_meta))
        metadata[:,0] = traj
        metadata[:,1] = lengths
        metadata[:,2:] = feature_means
        meta_labels = ['traj', 'traj_length'] + list(obs_keys)

    return metadata, meta_labels

def load_data(
        filepath,
        obs_keys=[
            'velocity',
            'jerk',
            'timegap',
            'time_to_collision',
            'lidar_1',
            'lidar_2',
            'rangerate_lidar_1',
            'rangerate_lidar_2'
        ],
        act_keys=[
            'accel',
        ],
        debug_size=None,
        y_key='beh_lon_T',
        y_1_val=1.,
        load_y=True,
        train_split=.8,
        mode='artificial',
        min_length=0,
        normalize_data=True,
        shuffle=False):
    
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names(filepath, mode)

    # load metadata
    metadata, meta_labels = load_metadata(filepath, mode, obs_keys)
    
    # optionally keep it to a reasonable size
    if debug_size is not None:
        x = x[:debug_size]
        metadata = metadata[:debug_size]

    if shuffle:
        idxs = np.random.permutation(len(x))
        x = x[idxs]
        metadata = metadata[idxs]

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # only keep samples with length > min_length
    valid_idxs = np.where(lengths > min_length)[0]
    x = x[valid_idxs]
    lengths = lengths[valid_idxs]
    metadata = metadata[valid_idxs]
    
    # y might not exist, so only laod it optionally
    if load_y:
        y_idx = np.where(feature_names == y_key)[0]
        y = np.zeros(len(x))
        one_idxs = np.where(x[:,-1,y_idx] == y_1_val)[0]
        y[one_idxs] = 1
    else:
        y = None
    
    # subselect relevant keys
    obs_idxs = [i for (i,n) in enumerate(feature_names) if n in obs_keys]
    obs = x[:,:,obs_idxs]
    act_idxs = [i for (i,n) in enumerate(feature_names) if n in act_keys]
    act = x[:,:,act_idxs]
    
    # train val split
    tidx = int(train_split *  len(obs))
    
    # val
    val_obs = obs[tidx:]
    val_act = act[tidx:]
    val_lengths = lengths[tidx:]
    val_metadata = metadata[tidx:]
    
    # train
    obs = obs[:tidx]
    act = act[:tidx]
    lengths = lengths[:tidx]
    metadata = metadata[:tidx]

    # normalize
    if normalize_data:
        info = normalize(obs, lengths)
        obs = info['x']
        val_obs = apply_normalize(val_obs, info['mean'], info['std'], val_lengths)

        info = normalize(act, lengths)
        act = info['x']
        val_act = apply_normalize(val_act, info['mean'], info['std'], val_lengths)

    if load_y:
        val_y = y[tidx:]
        y = y[:tidx]
    else:
        val_y = None

    # extract some common useful quantities
    n_samples, max_len, obs_dim = obs.shape
    act_dim = act.shape[-1]

    return dict(
        obs=obs,
        act=act,
        y=y,
        obs_keys=obs_keys,
        act_keys=act_keys,
        lengths=lengths,
        val_obs=val_obs,
        val_act=val_act,
        val_lengths=val_lengths,
        val_y=val_y,
        max_len=max_len,
        obs_dim=obs_dim,
        act_dim=act_dim,
        metadata=metadata,
        val_metadata=val_metadata,
        meta_labels=meta_labels
    )
