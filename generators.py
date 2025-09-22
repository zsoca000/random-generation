import numpy as np
import json
import matplotlib.pyplot as plt
import os.path as osp
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class BaselineMultisine:

    def __init__(self,config_path,seed=42):
        self.seed = seed
        self.params = self.load_params(config_path)
        self.rng = np.random.default_rng(seed)
        

    def load_params(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)


    def pick(self,param,size=1):
        if isinstance(param, list):
            return self.rng.uniform(param[0], param[1], size=size)
        return param

    
    def generate_t(self):
        dt = float(self.pick(self.params['general']['dt']))
        t_end = float(self.pick(self.params['general']['t_end']))
        return np.arange(0, t_end, dt)


    def generate_baseline(self,t):
        t_step = self.pick(self.params['baseline']['t_step'])
        step_times = np.arange(0, t[-1] + t_step, t_step)
        step_amplitudes = self.pick(self.params['baseline']['amplitude'],size=len(step_times))
        return np.interp(t, step_times, step_amplitudes)

    
    def generate_multisine(self, t):
        
        freq_min = float(self.pick(self.params['multisine']['freq_min']))
        freq_max = float(self.pick(self.params['multisine']['freq_max']))
        n_freqs = int(self.pick(self.params['multisine']['freq_max']))
        amplitudes = self.pick(self.params['multisine']['amplitude'],size=n_freqs)
        phases = self.pick(self.params['multisine']['phase'],size=n_freqs)

        freqs = np.linspace(freq_min, freq_max, n_freqs)

        multisine = np.zeros_like(t)
        for i in range(n_freqs):
            multisine += amplitudes[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])

        return multisine

    @property
    def signal(self):
        t = self.generate_t()
        baseline = self.generate_baseline(t)
        multisine = self.generate_multisine(t)
        sgn = baseline + multisine
        return sgn - sgn[0]


    def save_signals(self,n,folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path,f'bl_ms_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)],dtype=object),
            allow_pickle=True
        )
            

class PowerDecayMultisine:
    
    def __init__(self,config_path,seed=42):
        self.seed = seed
        self.params = self.load_params(config_path)
        self.rng = np.random.default_rng(seed)
        
    
    def load_params(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)


    @property
    def signal(self): 

        T = self.params['len_seq']
        N = self.params['n_freqs']
        p = self.params['power']
        u_max = self.params['amplitude']

        k = np.linspace(0, T, T)  
        a = self.rng.standard_normal(N)
        u = np.zeros_like(k)
        for n in range(1, N + 1):
            u += (a[n - 1] / n**p) * np.sin(2*np.pi*n*k/N)
        u = np.array(u)
        u = u/(np.max(u) - np.min(u))*u_max
        return u
    

    def save_signals(self,n,folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path,f'pd_ms_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)],dtype=object),
            allow_pickle=True
        )


class GaussianProcess:
    def __init__(self, config_path, seed=42):
        self.seed = seed
        self.params = self.load_params(config_path)
        self.rng = np.random.default_rng(seed)

    
    def load_params(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    
    def get_random_points(self):
        n_cp = int(self.params['n_cp'])
        amplitude = float(self.params['amplitude'])
        y = self.rng.uniform(-amplitude, amplitude, size=n_cp)
        X = np.arange(n_cp).reshape(-1, 1)
        return  X, y

    @property
    def signal(self):
        # Get random point
        X, y= self.get_random_points()
        
        # Fit GP
        kernel = RBF(length_scale=1.0) # + WhiteKernel(noise_level=1e-5)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(X, y)

        # Predict == interpolate points
        n_ts = int(self.params['n_ts'])
        X_pred = np.linspace(X.min(), X.max(), n_ts).reshape(-1, 1)
        y_pred, _ = gpr.predict(X_pred, return_std=True)

        return y_pred - y_pred[0]

    def save_signals(self,n,folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path,f'gp_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)],dtype=object),
            allow_pickle=True
        )


class RandomWalk:
    def __init__(self, config_path, seed=42):
        self.seed = seed
        self.params = self.load_params(config_path)
        self.rng = np.random.default_rng(seed)

    def load_params(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    @property
    def signal(self):
        n_steps = int(self.params['n_steps'])
        deps_max = float(self.params['deps_max'])      # max change per step
        smooth_kernel = float(self.params['smooth_kernel'])  # smoothing sigma

        # 1. Draw random increments
        d_eps = self.rng.uniform(-0.5, 0.5, size=n_steps)

        # 2. Envelope term every 10 increments
        n_env = n_steps // 10 + 1
        log_delta = self.rng.uniform(-3, -1, size=n_env)
        delta = 10 ** log_delta
        envelope = np.repeat(delta, 10)[:n_steps]
        d_eps = d_eps * envelope

        # 3. Capped random walk
        eps = np.zeros(n_steps)
        for t in range(1, n_steps):
            eps[t] = np.clip(
                eps[t-1] + d_eps[t],
                eps[t-1] - deps_max,
                eps[t-1] + deps_max
            )

        # 4. Smooth with Gaussian filter
        from scipy.ndimage import gaussian_filter1d
        eps_smooth = gaussian_filter1d(eps, sigma=smooth_kernel)

        return eps_smooth - eps_smooth[0]

    def save_signals(self, n, folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path, f'rw_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)], dtype=object),
            allow_pickle=True
        )


def get_generator(name,seed,config_path='configs'):
    if name == 'bl_ms':
        return BaselineMultisine(config_path=f'{config_path}/{name}_config.json',seed=seed)
    elif name == 'pd_ms':
        return PowerDecayMultisine(config_path=f'{config_path}/{name}_config.json',seed=seed)
    elif name == 'gp':
        return GaussianProcess(config_path=f'{config_path}/{name}_config.json',seed=seed)
    elif name == 'rw':
        return RandomWalk(config_path=f'{config_path}/{name}_config.json',seed=seed)
    else:
        raise ValueError(f"Unknown generator name: {name}")


class DataSet:
    
    def __init__(self,name,seed,n,folder_path):    
        self.folder_path = folder_path
        self.name = f'{name}_{seed}_{n}'
        self.seed = seed
        self.n = n
        self.data_path = osp.join(folder_path,f'{self.name}.npy')
        self.u_list = np.load(self.data_path, allow_pickle=True)
        self.lengths = [len(u) for u in self.u_list]

    def print_summary(self,lens=True,points=True):
        
        print(f"Data path: {self.data_path}")
        print(f"# of sequencnes: {len(self.u_list)}")
        print()
        
        if lens:
            print(f"Lengths:")
            print(f"\t* Min: {self.len_min}")
            print(f"\t* Max: {self.len_max}")
            print(f"\t* Mean : {self.len_mean:.2f}")
            print(f"\t* Std: {self.len_std:.2f}")
            print()
        
        if points:
            print(f"Data:")
            print(f"\t* Min: {self.data_min:.2f}")
            print(f"\t* Max: {self.data_max:.2f}")
            print(f"\t* Mean : {self.data_mean:.2f}")
            print(f"\t* Std: {self.data_std:.2f}")
            print()
            print()


    @property
    def len_min(self):
        return np.min(self.lengths)

    @property
    def len_max(self):
        return np.max(self.lengths)
    
    @property
    def len_mean(self):
        return np.mean(self.lengths)

    @property
    def len_std(self):
        return np.std(self.lengths) 
    
    @property
    def data_min(self):
        return np.concatenate(self.u_list).min()
    
    @property
    def data_max(self):
        return np.concatenate(self.u_list).max()

    @property
    def data_mean(self):
        return np.concatenate(self.u_list).mean()

    @property
    def data_std(self):
        return np.concatenate(self.u_list).std()


    def plot_samples(self,ax,num_samples=5):
        for i,u in enumerate(self.u_list[:num_samples]):
            ax.plot(u,label=f'Sample {i+1}')
        # ax.legend()
        ax.grid()


def comp_datasets_meta(data_set_list: list[DataSet]):
    
    fig, ax = plt.subplots(2,figsize=(8, 8), dpi=150,sharex=True)

    ax[0].boxplot(
        [data_set.lengths for data_set in data_set_list], 
        labels=[data_set.name for data_set in data_set_list], 
    )
    ax[0].set_ylabel('Sequence Length')

    ax[1].boxplot(
        [np.concatenate(data_set.u_list) for data_set in data_set_list], 
        labels=[data_set.name for data_set in data_set_list], 
    )
    ax[1].set_ylabel('Data points')


def comp_datasets(data_set_list: list[DataSet],num_samples=5):
    num = len(data_set_list)
    fig, ax = plt.subplots(num,figsize=(8,3*num), dpi=150,sharex=True,sharey=True)

    for i in range(num):
        data_set_list[i].plot_samples(ax=ax[i],num_samples=num_samples)
        ax[i].set_ylabel(f'{data_set_list[i].name}')
    ax[-1].set_xlabel('Time step')
    





