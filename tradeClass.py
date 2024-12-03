import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, binned_statistic_2d
import re
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import re
from tqdm import tqdm  

from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KernelDensity


import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import networkx as nx

import matplotlib.patches as patches
import random


class TradeAnalysis:
    def __init__(self, df, name_x, name_y):
        """
        Initialize the TradeAnalysis object and automatically perform initial analyses.
        """
        self.name_x = name_x
        self.name_y = name_y
        self.name_x_to_y = f'{name_x} to {name_y}'
        self.name_y_to_x = f'{name_y} to {name_x}'
        self.attribute_x_to_y = f'x_to_y'
        self.attribute_y_to_x = f'y_to_x'
        
        # Data extraction
        self.extract_trade_volumes(df, name_x, name_y)
        
        # Initial computations
        self.compute_initial_analyses()


    def extract_trade_volumes(self, df, name_x, name_y):
        """
        Extracts and aligns trade volumes from the dataframe based on country names,
        dropping days where only one country traded with the other and keeping days where both traded.
        """
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2))

        # Filtering data for both trade directions and setting 'Date' as index
        x_to_y = df[(df['Export_country'] == name_x) & (df['Import_country'] == name_y)].set_index('Date')['Value (SUM)']
        y_to_x = df[(df['Export_country'] == name_y) & (df['Import_country'] == name_x)].set_index('Date')['Value (SUM)']

        self.length_x_to_y = len(x_to_y)
        self.length_y_to_x = len(y_to_x)

        trade_data = pd.DataFrame({'x_to_y': x_to_y, 'y_to_x': y_to_x})

        trade_data.dropna(inplace=True)
        trade_data.reset_index(inplace=True)

        self.data_x_to_y = trade_data['x_to_y'].values
        self.data_y_to_x = trade_data['y_to_x'].values
        self.dates = trade_data['Date'].values 


    def compute_initial_analyses(self):
        """
        Perform initial analyses such as computeing optimal bins and KDE bandwidth.
        """

        self.ecdf_x_to_y, self.eccdf_x_to_y = self.compute_ecdf_ccdf(self.data_x_to_y)
        self.ecdf_y_to_x, self.eccdf_y_to_x = self.compute_ecdf_ccdf(self.data_y_to_x)

        self.optimal_bins_x_to_y = self.compute_optimal_bins(self.data_x_to_y)
        self.optimal_bins_y_to_x = self.compute_optimal_bins(self.data_y_to_x)
        
        self.best_kde_x_to_y, self.best_bandwidth_x_to_y = self.compute_best_bandwidth_kde(self.data_x_to_y)
        self.best_kde_y_to_x, self.best_bandwidth_y_to_x = self.compute_best_bandwidth_kde(self.data_y_to_x)
        
        self.kde_cdf_x_to_y = self.compute_kde_cdf(self.data_x_to_y, self.best_kde_x_to_y)
        self.kde_cdf_y_to_x = self.compute_kde_cdf(self.data_y_to_x, self.best_kde_y_to_x)

        self.best_joint_bandwidth, self.joint_kde = self.find_best_bandwidth_joint_kde(self.data_x_to_y, self.data_y_to_x)
        self.optimal_bins_x, self.optimal_bins_y, self.joint_hist = self.find_optimal_bins_joint_histogram(self.data_x_to_y, self.data_y_to_x)

        # shannon entropy estimate
        self.shannon_entropy_joint = self.calculate_joint_entropy(self.data_x_to_y, self.data_y_to_x)
        self.shannon_entropy_x_to_y = self.calculate_marginal_entropy(self.data_x_to_y, self.optimal_bins_x_to_y)
        self.shannon_entropy_y_to_x = self.calculate_marginal_entropy(self.data_y_to_x, self.optimal_bins_y_to_x)

        # mutual information
        self.pearson_corr, self.linear_mi = self.calculate_pearson()
        self.mutual_info = self.calculate_mi(self.data_x_to_y, self.data_y_to_x)


    def metrics(self):
        # Print the results with rounding to 5 significant figures
        print(f"Mutual Information: {self.mutual_info:.5g}")
        print(f"Pearson Correlation Mutual Information: {self.pearson_corr:.5g}")

        print(f"Number of observations from {self.name_x_to_y}: {self.length_x_to_y}")
        print(f"Number of observations from {self.name_y_to_x}: {self.length_y_to_x}")
        print(f"Optimal Bins for {self.name_x_to_y}: {self.optimal_bins_x_to_y:.5g}")
        print(f"Optimal Bins for {self.name_y_to_x}: {self.optimal_bins_y_to_x:.5g}")
        print(f"Optimal BW of {self.name_x_to_y}: {self.best_bandwidth_x_to_y:.5g}")
        print(f"Optimal BW of {self.name_y_to_x}: {self.best_bandwidth_y_to_x:.5g}")

    def return_data(self):
        return pd.DataFrame({'x_to_y': self.data_x_to_y, 'y_to_x': self.data_y_to_x})


    def plot_call(self):
        self.plot_rank_frequency()
        self.plot_kde_hist()
        self.plot_pp()
        self.plot_joint_kde()
        self.plot_scatter()


    def calculate_entropy(self, probabilities):
        """Calculate Shannon entropy for given probabilities."""
        probabilities = probabilities[probabilities > 0]  
        return -np.sum(probabilities * np.log2(probabilities))

    def calculate_joint_entropy(self, X, Y):
        """Calculate joint entropy for two variables X and Y."""
        data, _, _ = np.histogram2d(X, Y, bins=[self.optimal_bins_x, self.optimal_bins_y], density=True)
        joint_prob = data / np.sum(data)  
        return self.calculate_entropy(joint_prob.flatten())

    def calculate_marginal_entropy(self, X, bins):
        """Calculate marginal entropy for variable X."""
        data, _ = np.histogram(X, bins=bins, density=True)
        probabilities = data / np.sum(data)
        return self.calculate_entropy(probabilities)

    def calculate_mi(self, X, Y):
        """Calculate mutual information between variables X and Y."""
        entropy_X = self.calculate_marginal_entropy(X, self.optimal_bins_x_to_y )
        entropy_Y = self.calculate_marginal_entropy(Y, self.optimal_bins_y_to_x)
        joint_entropy = self.calculate_joint_entropy(X, Y)
        mi = entropy_X + entropy_Y - joint_entropy
        return mi



    def compute_ecdf_ccdf(self, data):
        sorted_data = np.sort(data)
        q = len(sorted_data)
        ranks = np.arange(1, q + 1)

        cdf_estimator = ranks / (q + 1)
        ccdf_estimator = 1 - cdf_estimator

        return cdf_estimator, ccdf_estimator


    def compute_optimal_bins(self, data):
        """
        Compute the optimal number of bins for the given data.
        """
        if len(data) == 0:  
            return None 
        
        ss = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
        scores_for_bins = {n_bins: [] for n_bins in range(5, 70)}
        
        for n_bins in range(5, 30, 1):
            for train_index, test_index in ss.split(data):
                train_data, test_data = data[train_index], data[test_index]
                
                if len(train_data) == 0 or len(test_data) == 0:  
                    continue
                
                hist, bin_edges = np.histogram(train_data, bins=n_bins, density=True)
                if np.any(hist == 0) or len(bin_edges) < 2:  
                    continue
                
                bin_width = bin_edges[1] - bin_edges[0]  
                kde = KernelDensity(bandwidth=bin_width, kernel='gaussian')
                kde.fit(train_data.reshape(-1, 1))
                
                if len(test_data) > 0:
                    test_scores = kde.score_samples(test_data.reshape(-1, 1))
                    scores_for_bins[n_bins].append(np.sum(test_scores))
        

        avg_scores = {n_bins: np.mean(scores) if len(scores) > 0 else float('-inf') for n_bins, scores in scores_for_bins.items()}
        optimal_bins = max(avg_scores, key=avg_scores.get) if avg_scores else None

        return optimal_bins


    def compute_best_bandwidth_kde(self, data):
        if len(data) == 0:
            return None, None  

        n_splits = 10
        test_size = 0.3
        rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        bandwidths = np.linspace(start=0.01, stop=1.0, num=30)
        best_bandwidths = []
        
        for train_index, test_index in rs.split(data):
            best_log_likelihood = -np.inf
            for bw in bandwidths:
                training_data = data[train_index]
                validation_data = data[test_index]
                if len(training_data) == 0 or len(validation_data) == 0:
                    continue  
                kde = gaussian_kde(training_data, bw_method=bw)
                validation_pdf = kde(validation_data)
                epsilon = 1e-12 
                log_likelihood = np.sum(np.log(validation_pdf + epsilon))
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_bw = bw
            if 'best_bw' in locals():  
                best_bandwidths.append(best_bw)

        if len(best_bandwidths) == 0:
            return None, None  
        
        best_bandwidth = np.median(best_bandwidths)
        best_kde = gaussian_kde(data, bw_method=best_bandwidth) if len(data) > 0 else None
        
        return best_kde, best_bandwidth

    
    def find_optimal_bins_joint_histogram(self, data1, data2):
        """
        Finds the optimal number of bins for a joint histogram of two datasets.
        
        Parameters:
        - attr1, attr2: str, attribute names of the datasets for joint histogram (e.g., 'data_x_to_y', 'data_y_to_x').
        
        Returns:
        - optimal_bins_x: int, the optimal number of bins for the first dataset.
        - optimal_bins_y: int, the optimal number of bins for the second dataset.
        """
        if len(data1) == 0 or len(data2) == 0:
            return None, None

        bin_range_x = np.linspace(5, 25, 1).astype(int)
        bin_range_y = np.linspace(5, 25, 1).astype(int)

        best_score = np.inf
        optimal_bins_x, optimal_bins_y = 5, 5 

        for bins_x in bin_range_x:
            for bins_y in bin_range_y:
                hist, x_edges, y_edges, _ = binned_statistic_2d(data1, data2, None, 'count', bins=[bins_x, bins_y])
            
                score = -np.sum(hist)
                if score < best_score:
                    best_score = score
                    optimal_bins_x, optimal_bins_y = bins_x, bins_y
        
        hist_joint = binned_statistic_2d(data1, data2, None, 'count', bins=[optimal_bins_x, optimal_bins_y])
        return optimal_bins_x, optimal_bins_y, hist_joint


    def compute_kde_cdf(self, data, kde):
        """
        Compute the CDF using the given KDE model.
        """
        start = np.min(data)
        end = np.max(data)
        x = np.linspace(start, end, 1000)

        pdf = kde.evaluate(x)
        kde_cdf = np.cumsum(pdf) * (end - start) / 1000

        return  kde_cdf 

    
    def find_best_bandwidth_joint_kde(self, data1, data2):
        """
        Finds the best bandwidth for joint KDE estimation of two datasets using cross-validation.
        
        Parameters:
        - attr1, attr2: str, attribute names of the datasets for joint KDE (e.g., 'data_x_to_y', 'data_y_to_x').
        
        Returns:
        - best_bandwidth: float, the best bandwidth found for joint KDE.
        """

        if len(data1) == 0 or len(data2) == 0:
            return None

        n_splits = 10
        test_size = 0.3
        rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        bandwidths = np.linspace(start=0.01, stop=1.0, num=10)  
        best_bandwidths = []

        for train_index, test_index in rs.split(data1):
            train_data1 = data1[train_index]
            train_data2 = data2[train_index]
            validation_data1 = data1[test_index]
            validation_data2 = data2[test_index]

            train_data = np.vstack([train_data1, train_data2]).T
            validation_data = np.vstack([validation_data1, validation_data2]).T

            best_log_likelihood = -np.inf
            for bw in bandwidths:
                kde = gaussian_kde(train_data.T, bw_method=bw)
                log_likelihood = np.sum(kde.logpdf(validation_data.T))
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_bw = bw
            best_bandwidths.append(best_bw)

        best_bandwidth = np.median(best_bandwidths)

        joint_data = np.vstack([data1, data2]).T
        best_kde = gaussian_kde(joint_data.T, bw_method=best_bandwidth)
        
        return best_bandwidth , best_kde


    def calculate_pearson(self):
        pearson_corr = np.corrcoef(self.data_x_to_y, self.data_y_to_x)[0, 1]
        lin_MI = -0.5 * np.log(1 - pearson_corr ** 2)
        return pearson_corr, lin_MI


    def plot_rank_frequency(self):
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))

        for i, (data, name) in enumerate(zip([self.data_x_to_y, self.data_y_to_x], [self.name_x_to_y, self.name_y_to_x])):
            sorted_data = np.sort(data)
            q = len(sorted_data)
            ranks = np.arange(1, q + 1)
            cdf_estimator = ranks / (q + 1)
            ccdf_estimator = 1 - cdf_estimator
            epsilon = np.sqrt(np.log(2 / 0.1) / (2 * q))

            axs[i].step(sorted_data, cdf_estimator, where='post', label=f'Empirical CDF for {name}', color='steelblue')
            axs[i].fill_between(sorted_data, np.maximum(cdf_estimator - epsilon, 0), np.minimum(cdf_estimator + epsilon, 1),
                                alpha=0.2, color='gray', label='DKW 90% Confidence Interval')
            
            axs[i].set_title(f'Rank-Frequency Plot for Exports from {name}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Cumulative Probability')
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_kde_hist(self):
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))

        for i, (data, name, attr) in enumerate(zip([self.data_x_to_y, self.data_y_to_x], [self.name_x_to_y, self.name_y_to_x], [self.attribute_x_to_y, self.attribute_y_to_x])):
            optimal_bins = getattr(self, f'optimal_bins_{attr}')
            best_kde = getattr(self, f'best_kde_{attr}')
            best_bandwidth = getattr(self, f'best_bandwidth_{attr}')

            x_d = np.linspace(data.min(), data.max(), 1000)
            axs[i].hist(data, bins=optimal_bins, density=True, alpha=0.5, label=f'Histogram\n(bins={optimal_bins})', color='steelblue', edgecolor='black')
            axs[i].plot(x_d, best_kde(x_d), label=f'KDE (bw={best_bandwidth:.2f})', color='black', linewidth=2)
            
            axs[i].set_title(f'Optimal KDE for {name}', fontsize=16)
            axs[i].set_xlabel('Trade Volume (Value)', fontsize=14)
            axs[i].set_ylabel('Density', fontsize=14)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].legend(fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_pp(self):
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))

        for i, (data, name, attr) in enumerate(zip([self.data_x_to_y, self.data_y_to_x], [self.name_x_to_y, self.name_y_to_x], [self.attribute_x_to_y, self.attribute_y_to_x])):
            kde_cdf = getattr(self, f'kde_cdf_{attr}')

            start = np.min(data)
            end = np.max(data)
            x = np.linspace(start, end, 1000)

            kde_cdf_interp = np.interp(sorted(data), x, kde_cdf)
            ecdf = getattr(self, f'ecdf_{attr}')

            axs[i].plot(ecdf, kde_cdf_interp, linestyle='none', marker='o', markersize=5, markeredgecolor='black', markerfacecolor='steelblue')
            axs[i].plot([0, 1], [0, 1], 'k--', label='45-degree line')  # Diagonal line for reference
            
            axs[i].set_title(f'P-P Plot of Empirical CDF vs. KDE CDF for {name}', fontsize=16)
            axs[i].set_xlabel('Empirical CDF', fontsize=14)
            axs[i].set_ylabel('KDE CDF', fontsize=14)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].legend(fontsize=12)
            axs[i].grid(True, which='both', linestyle='--', linewidth=0.5, color='black')

        plt.tight_layout()
        plt.show()


    def plot_joint_kde(self):
        """
        Plots the joint KDE for two datasets using the provided KDE model.
        """
        data1 = self.data_x_to_y
        data2 = self.data_y_to_x
        best_kde = self.joint_kde

        if best_kde is None:
            print("No KDE model provided.")
            return
        
        x_min, x_max = data1.min(), data1.max()
        y_min, y_max = data2.min(), data2.max()
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        
        Z = np.reshape(best_kde(grid_coords).T, X.shape)
        
        plt.figure(figsize=(18, 6))
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='Blues')
        plt.colorbar(label='Probability Density')
        plt.scatter(data1, data2, s=2, facecolor='white', edgecolor='none', alpha=0.5)  
        plt.title('Joint KDE')
        plt.xlabel({self.name_x_to_y})
        plt.ylabel({self.name_y_to_x})
        plt.show()

    def plot_scatter(self):
        """
        Plots a scatter plot for two datasets.

        Parameters:
        - None, but utilizes the class attributes self.data_x_to_y and self.data_y_to_x for plotting.
        """
        data1 = self.data_x_to_y
        data2 = self.data_y_to_x

        if data1 is None or data2 is None:
            print("Data is missing.")
            return
        
        plt.figure(figsize=(6, 6))
        
        plt.scatter(data1, data2, s=50, color='steelblue', alpha=0.8, label='Data Points')
        
        plt.title('Scatter Plot between Data1 and Data2', fontsize=16)
        plt.xlabel({self.name_x_to_y})
        plt.ylabel({self.name_y_to_x})
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
        
        plt.show()
