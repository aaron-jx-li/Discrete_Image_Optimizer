import numpy as np
import matplotlib.pylab as plt
import torch
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import euclidean_distances

class RandomOptimizer_Batch:    
    def __init__(self, brain_rewards, num_stimuli, N_init=50, batch_size=50, noise_scale=0.1, shuffle=True):
        self.num_stimuli = num_stimuli
        self.brain_rewards_noisy = brain_rewards + np.random.normal(0.0, brain_rewards*0.1, brain_rewards.shape)
        if shuffle:
            self.chosen_indices = np.random.choice(self.num_stimuli, N_init)
        else:    
            self.chosen_indices = np.arange(N_init)
        self.init_indices = self.chosen_indices
        self.batch_size = batch_size
        
        init_rewards = np.take(self.brain_rewards_noisy, self.init_indices, axis=0)
        init_mean_rewards = np.mean(np.array(init_rewards), axis=1)
        
        self.best_index = np.argmax(init_mean_rewards)
        self.best_reward = self.brain_rewards_noisy[self.best_index]
        
    def propose_batch(self):
        num_seen_stimuli = len(np.unique(self.chosen_indices))
        #if num_seen_stimuli >= self.num_stimuli:
        #    print(num_seen_stimuli, self.num_stimuli)
        #    print("All images have been proposed at least once!")
        indices = np.random.choice(self.num_stimuli, self.batch_size)
        return indices
    
    def evaluate(self, indices):
        self.chosen_indices = np.concatenate((self.chosen_indices, indices))
        #print("Number of proposed images: ", len(self.chosen_indices))
        
        rewards = np.take(self.brain_rewards_noisy, indices, axis=0)
        mean_rewards = np.mean(np.array(rewards), axis=1)
        
        max_index = np.argmax(mean_rewards)
        if mean_rewards[max_index] > np.mean(np.array(self.best_reward)):
            self.best_index = indices[max_index]
            self.best_reward = self.brain_rewards_noisy[self.best_index]
        return np.take(self.brain_rewards_noisy, indices, axis=0)
    
    def propose_one(self, candidates, mode):
        candidate = np.random.choice(candidates)
        if mode == 'naive random':
            candidate_rewards = np.take(self.brain_rewards_noisy, candidates, axis=0)
            distribution = np.mean(np.array(candidate_rewards), axis=1)
            distribution = distribution / np.sum(distribution)
            candidate = np.random.choice(candidates, p=distribution)
        elif mode == 'random decay':
            return
        return candidate, self.brain_rewards_noisy[candidate]
        