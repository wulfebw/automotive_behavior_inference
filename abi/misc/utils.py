
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
    return lengths


def normalize(x, axes):
    mean = np.mean(x, axes)
    x = x - mean
    std = np.std(x, axes)
    x = x / std
    return dict(x=x, mean=mean, std=std)

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
        train_split=.8):
    
    f = h5py.File(filepath, 'r')
    x = f['risk/features']
    
    if debug_size is not None:
        x = x[:debug_size]
    # append a zero to the front of the timeseries dim of x
    # because the model assumes this occurs
    x = np.concatenate((np.zeros((x.shape[0],1,x.shape[2])), x), axis=1)
    feature_names=f['risk'].attrs['feature_names']
    
    y_idx = np.where(feature_names == y_key)[0]
    y = np.zeros(len(x))
    one_idxs = np.where(x[:,-1,y_idx] == y_1_val)[0]
    y[one_idxs] = 1
    
    obs_idxs = [i for (i,n) in enumerate(feature_names) if n in obs_keys]
    obs = x[:,:,obs_idxs]
    obs = normalize(obs, (0,1))['x']
    act_idxs = [i for (i,n) in enumerate(feature_names) if n in act_keys]
    act = x[:,:,act_idxs]
    act = normalize(act, (0,1))['x']
    
    n_samples, max_len, obs_dim = obs.shape
    max_len = max_len - 1 
    act_dim = act.shape[-1]
    lengths = np.ones(n_samples) * max_len
    
    # train val split
    tidx = int(train_split *  n_samples)
    val_obs = obs[tidx:]
    val_act = act[tidx:]
    val_lengths = lengths[tidx:]
    val_y = y[tidx:]
    obs = obs[:tidx]
    act = act[:tidx]
    lengths = lengths[:tidx]
    y = y[:tidx]
    
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
        act_dim=act_dim
    )