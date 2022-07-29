import numpy as np
import matplotlib.pylab as plt
import torch
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

class AdeptOptimizer_Batch:
    def __init__(self, embeddings_pool, brain_rewards, N_init=50, h=200, shuffle=True, batch_size=50, 
                 noise_scale=0.01, decay=0.5, alpha=0):
        self.num_stimuli = embeddings_pool.shape[0]
        self.embed_dim = embeddings_pool.shape[1]
        self.kernel_mat = rbf_kernel(embeddings_pool, gamma=1/(h*h))
        self.reward_diffs = np.zeros((N_init, N_init))
        self.batch_size = batch_size
        self.decay = decay
        self.alpha = alpha
        # All rewards with noises are precomputed, which hopefully doesn't affect the effect of simulation
        # with decay update in 'propose_batch'
        self.brain_rewards_noisy = brain_rewards + np.random.normal(0.0, brain_rewards*0.1, brain_rewards.shape)
        # without any update
        self.brain_rewards_noisy_copy = np.copy(self.brain_rewards_noisy)
        
        
        if shuffle:
            self.chosen_indices = np.random.choice(self.num_stimuli, N_init)
            self.reward_diffs += euclidean_distances(np.take(self.brain_rewards_noisy, self.chosen_indices, axis=0))
        else:    
            self.reward_diffs += euclidean_distances(self.brain_rewards_noisy[:N_init])
            self.chosen_indices = np.arange(N_init)
        self.init_indices = self.chosen_indices
        self.reward_norms = np.linalg.norm(self.brain_rewards_noisy, axis=1)
        
        init_rewards = np.take(self.brain_rewards_noisy, self.init_indices, axis=0)
        init_mean_rewards = np.mean(np.array(init_rewards), axis=1)
        
        self.best_index = np.argmax(init_mean_rewards)
        self.best_reward = self.brain_rewards_noisy[self.best_index]
        print("Initial Images #: ", self.chosen_indices)
        
    #def add_noise(self, x, scale=0.01):
    #    return x + np.random.normal(0.0, scale, x.shape)
    
    def propose_batch(self):
        #if len(self.chosen_indices) >= len(images_pool):
        #    print("All images have been proposed.")
        #    return
        num_seen_stimuli = len(np.unique(self.chosen_indices))
        if num_seen_stimuli >= self.num_stimuli:
            print("All images have been proposed at least once!")
            return
        candidates = np.zeros(self.num_stimuli)
        for i in range(self.num_stimuli):
            kernel_embedding = self.kernel_mat[self.chosen_indices][:, i] / np.sum(self.kernel_mat[self.chosen_indices][:, i])
            norm = np.dot(kernel_embedding, np.take(self.reward_norms, self.chosen_indices))
            avg_dist = np.mean(np.matmul(kernel_embedding.T, self.reward_diffs))
            candidates[i] = norm + self.alpha * avg_dist
                
        best_candidates = np.argsort(candidates)[-self.batch_size:]
        return best_candidates
    
    def evaluate_and_update_batch(self, best_candidates):
        #dist = euclidean_distances(np.take(self.brain_rewards_noisy_copy, self.chosen_indices, axis=0), self.brain_rewards_noisy[best_candidates])
        for i in best_candidates:
            self.brain_rewards_noisy[i] = self.brain_rewards_noisy[i]*self.decay
        self.reward_diffs = np.hstack((self.reward_diffs, euclidean_distances(np.take(self.brain_rewards_noisy_copy, self.chosen_indices, axis=0), self.brain_rewards_noisy_copy[best_candidates])))
        self.chosen_indices = np.concatenate((self.chosen_indices, best_candidates))
        self.reward_diffs = np.vstack((self.reward_diffs, np.transpose(euclidean_distances(np.take(self.brain_rewards_noisy_copy, self.chosen_indices, axis=0), self.brain_rewards_noisy_copy[best_candidates]))))
        
        best_reward = self.brain_rewards_noisy_copy[best_candidates[-1]]
        if np.mean(np.array(best_reward)) > np.mean(np.array(self.best_reward)):
            self.best_index = best_candidates[-1]
            self.best_reward = best_reward
        #print("Number of proposed images: ", len(self.chosen_indices))
        #print("Image #: ", best_candidates)
        return np.take(self.brain_rewards_noisy_copy, best_candidates, axis=0)
    
    def propose_one(self, candidates, mode):
        candidate = np.random.choice(candidates)
        if mode == 'naive random':
            candidate_rewards = np.take(self.brain_rewards_noisy_copy, candidates, axis=0)
            distribution = np.mean(np.array(candidate_rewards), axis=1)
            distribution = distribution / np.sum(distribution)
            candidate = np.random.choice(candidates, p=distribution)
        elif mode == 'random decay':
            candidate_rewards = np.take(self.brain_rewards_noisy, candidates, axis=0)
            distribution = np.mean(np.array(candidate_rewards), axis=1)
            distribution = distribution / np.sum(distribution)
            candidate = np.random.choice(candidates, p=distribution)
        elif mode == 'softmax':
            candidate_rewards = np.take(self.brain_rewards_noisy_copy, candidates, axis=0)
            distribution = np.mean(np.array(candidate_rewards), axis=1)
            distribution = np.exp(distribution) / np.sum(np.exp(distribution))
            candidate = np.random.choice(candidates, p=distribution)
        return candidate, self.brain_rewards_noisy_copy[candidate]
    
    
    
    
    
    
    
    