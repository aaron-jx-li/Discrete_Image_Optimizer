import numpy as np
import matplotlib.pylab as plt
import torch
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import euclidean_distances


class AdeptOptimizer_Single:
    def __init__(self, images_pool, embeddings_pool, brain_rewards, N_init=50, h=200, shuffle=True):
        self.num_stimuli = images_pool.shape[0]
        self.embed_dim = embeddings_pool.shape[1]
        self.kernel_mat = rbf_kernel(embeddings_pool, gamma=1/(h*h))
        self.reward_diffs = np.zeros((N_init, N_init))
        if shuffle:
            self.chosen_indices = np.random.choice(self.num_stimuli, N_init)
            self.reward_diffs += euclidean_distances(np.take(brain_rewards, self.chosen_indices, axis=0))
        else:    
            self.reward_diffs += euclidean_distances(brain_rewards[:N_init])
            self.chosen_indices = np.arange(N_init)
        
        self.reward_norms = np.linalg.norm(brain_rewards, axis=1)
        
    def update_and_propose(self):
        candidates = np.zeros(self.num_stimuli)
        for i in range(self.num_stimuli):
            if i not in self.chosen_indices:
                num_seen_stimuli = len(self.chosen_indices)
                kernel_embedding = self.kernel_mat[self.chosen_indices][:, i] / np.sum(self.kernel_mat[self.chosen_indices][:, i])
                norm = np.dot(kernel_embedding, np.take(self.reward_norms, self.chosen_indices))
                avg_dist = np.mean(np.matmul(kernel_embedding.T, self.reward_diffs))
                candidates[i] = norm + avg_dist
                
        best_candidate = np.argmax(candidates)
        dist = euclidean_distances(np.take(brain_rewards, self.chosen_indices, axis=0), brain_rewards[best_candidate].reshape(1, -1))
        self.reward_diffs = np.hstack((self.reward_diffs, euclidean_distances(np.take(brain_rewards, self.chosen_indices, axis=0), brain_rewards[best_candidate].reshape(1, -1))))
        self.chosen_indices = np.append(self.chosen_indices, best_candidate)
        self.reward_diffs = np.vstack((self.reward_diffs, np.transpose(euclidean_distances(np.take(brain_rewards, self.chosen_indices, axis=0), brain_rewards[best_candidate].reshape(1, -1)))))
        print("Number of proposed images: ", len(self.chosen_indices))
        return images_pool[best_candidate]