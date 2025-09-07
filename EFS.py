import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import make_classification
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
import math
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Replace two points (..) with the file path.


def create_pearson_correlation_map(data, threshold_type='median', custom_threshold=None, 
                                   figsize=(16, 14), save_plot=True, plot_title=None):
    """
    Create a Pearson's correlation coefficient correlation map with selectable thresholds.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset (features only, without target columns)
    threshold_type : str, default='median'
        Type of threshold to apply: 'median', 'log2', 'custom', or 'none'
    custom_threshold : float, optional
        Custom threshold value when threshold_type='custom'
    figsize : tuple, default=(16, 14)
        Figure size for the plot
    save_plot : bool, default=True
        Whether to save the plot as PNG file
    plot_title : str, optional
        Custom title for the plot
        
    Returns:
    --------
    correlation_matrix : pd.DataFrame
        Full correlation matrix
    filtered_correlation_matrix : pd.DataFrame
        Correlation matrix after applying threshold
    threshold_value : float
        The actual threshold value used
    """
    
    print(f"Creating Pearson correlation map with threshold type: {threshold_type}")
    
    # Calculate correlation matrix
    correlation_matrix = data.corr(method='pearson')
    
    # Determine threshold value
    if threshold_type == 'median':
        # Use median of absolute correlation values (excluding diagonal)
        abs_corr = np.abs(correlation_matrix.values)
        np.fill_diagonal(abs_corr, 0)  # Exclude diagonal
        threshold_value = np.median(abs_corr[abs_corr > 0])
        print(f"Using median threshold: {threshold_value:.4f}")
        
    elif threshold_type == 'log2':
        # Use log2 of number of features
        n_features = data.shape[1]
        threshold_value = math.log2(n_features)
        print(f"Using log2 threshold: {threshold_value:.4f} (log2({n_features}))")
        
    elif threshold_type == 'custom':
        if custom_threshold is None:
            raise ValueError("custom_threshold must be provided when threshold_type='custom'")
        threshold_value = custom_threshold
        print(f"Using custom threshold: {threshold_value:.4f}")
        
    elif threshold_type == 'none':
        threshold_value = 0
        print("No threshold applied (showing all correlations)")
        
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}. Use 'median', 'log2', 'custom', or 'none'")
    
    # Create filtered correlation matrix
    if threshold_type != 'none':
        # Apply threshold to absolute correlations
        abs_corr = np.abs(correlation_matrix.values)
        mask = abs_corr < threshold_value
        filtered_correlation_matrix = correlation_matrix.copy()
        filtered_correlation_matrix.values[mask] = 0
    else:
        filtered_correlation_matrix = correlation_matrix
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Full correlation matrix
    sns.heatmap(correlation_matrix, 
                annot=False,  # Don't show values to avoid overcrowding
                cmap='RdBu_r', 
                center=0,
                square=True,
                ax=ax1,
                cbar_kws={'label': 'Pearson Correlation'})
    ax1.set_title(f'Full Pearson Correlation Matrix\n({data.shape[1]} features)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Features')
    
    # Plot 2: Filtered correlation matrix
    if threshold_type != 'none':
        sns.heatmap(filtered_correlation_matrix, 
                    annot=False,  # Don't show values to avoid overcrowding
                    cmap='RdBu_r', 
                    center=0,
                    square=True,
                    ax=ax2,
                    cbar_kws={'label': 'Pearson Correlation'})
        ax2.set_title(f'Filtered Correlation Matrix\n(Threshold: {threshold_value:.4f})', 
                      fontsize=14, fontweight='bold')
    else:
        sns.heatmap(filtered_correlation_matrix, 
                    annot=False,
                    cmap='RdBu_r', 
                    center=0,
                    square=True,
                    ax=ax2,
                    cbar_kws={'label': 'Pearson Correlation'})
        ax2.set_title(f'Correlation Matrix\n(No threshold applied)', 
                      fontsize=14, fontweight='bold')
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Features')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add overall title
    if plot_title is None:
        plot_title = f"Pearson Correlation Analysis - {data.shape[1]} Features"
    
    fig.suptitle(plot_title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        filename = f"pearson_correlation_map_{threshold_type}_{data.shape[1]}features.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nCorrelation Analysis Summary:")
    print(f"Total features: {data.shape[1]}")
    print(f"Threshold type: {threshold_type}")
    print(f"Threshold value: {threshold_value:.4f}")
    
    if threshold_type != 'none':
        # Count non-zero correlations after filtering
        n_nonzero = np.count_nonzero(filtered_correlation_matrix.values)
        n_total = filtered_correlation_matrix.values.size
        print(f"Non-zero correlations after filtering: {n_nonzero}/{n_total} ({n_nonzero/n_total*100:.1f}%)")
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val >= threshold_value:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        
        print(f"\nTop 10 highly correlated feature pairs (≥{threshold_value:.4f}):")
        for i, pair in enumerate(high_corr_pairs[:10]):
            print(f"  {i+1:2d}. {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.4f}")
        
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs) - 10} more pairs")
    
    return correlation_matrix, filtered_correlation_matrix, threshold_value


def analyze_feature_correlations(data, target_col=None, top_n=20, figsize=(15, 10)):
    """
    Analyze feature correlations with target variable and between features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_col : str, optional
        Name of target column. If None, analyze only feature correlations
    top_n : int, default=20
        Number of top correlations to show
    figsize : tuple, default=(15, 10)
        Figure size for the plot
    """
    
    print(f"Analyzing feature correlations for {data.shape[1]} features...")
    
    if target_col is not None and target_col in data.columns:
        # Separate features and target
        features = data.drop(columns=[target_col])
        target = data[target_col]
        
        # Calculate correlations with target
        target_correlations = []
        for col in features.columns:
            corr, p_value = pearsonr(features[col], target)
            target_correlations.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': p_value
            })
        
        # Sort by absolute correlation
        target_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"\nTop {top_n} features correlated with target '{target_col}':")
        print(f"{'Feature':<30} {'Correlation':<12} {'Abs Corr':<12} {'P-value':<12}")
        print("-" * 70)
        
        for i, corr_info in enumerate(target_correlations[:top_n]):
            print(f"{corr_info['feature']:<30} {corr_info['correlation']:<12.4f} "
                  f"{corr_info['abs_correlation']:<12.4f} {corr_info['p_value']:<12.4e}")
        
        # Plot target correlations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Top correlations with target
        top_features = [corr_info['feature'] for corr_info in target_correlations[:top_n]]
        top_corrs = [corr_info['correlation'] for corr_info in target_correlations[:top_n]]
        colors = ['red' if c < 0 else 'blue' for c in top_corrs]
        
        bars = ax1.barh(range(len(top_features)), top_corrs, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features)
        ax1.set_xlabel('Correlation with Target')
        ax1.set_title(f'Top {top_n} Feature Correlations with Target')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, top_corrs)):
            ax1.text(bar.get_width() + (0.01 if corr >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', ha='left' if corr >= 0 else 'right', va='center')
        
        # Plot 2: Distribution of correlations
        all_corrs = [corr_info['correlation'] for corr_info in target_correlations]
        ax2.hist(all_corrs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero correlation')
        ax2.set_xlabel('Correlation with Target')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Feature-Target Correlations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save target correlations to CSV
        target_corr_df = pd.DataFrame(target_correlations)
        target_corr_df = target_corr_df.sort_values('abs_correlation', ascending=False)
        target_corr_df.to_csv('target_correlations.csv', index=False)
        print(f"\nTarget correlations saved to 'target_correlations.csv'")
        
        return target_correlations
    
    else:
        print("No target column specified. Analyzing only feature correlations.")
        return None


class EnsembleFeatureSelection:
    """
    Ensemble Feature Selection using multiple methods:
    - Pearson Correlation
    - Chi-squared Test
    - ANOVA F-test
    - ReliefF
    - Gain Ratio (Information Gain Ratio)
    """
    
    def __init__(self):
        self.feature_scores_ = {}
        self.selected_features_ = {}
        self.ensemble_features_ = None
        self.feature_names_ = None
        
    def _pearson_correlation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate Pearson correlation coefficient for each feature with target."""
        scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            try:
                # Check for valid data
                feature_data = X[:, i]
                if np.any(np.isnan(feature_data)) or np.any(np.isnan(y)):
                    scores[i] = 0.0
                    continue
                
                if np.var(feature_data) == 0 or np.var(y) == 0:
                    scores[i] = 0.0
                    continue
                
                corr, _ = pearsonr(feature_data, y)
                if corr is None or np.isnan(corr):
                    scores[i] = 0.0
                else:
                    scores[i] = abs(corr)  # Take absolute value for ranking
            except Exception as e:
                print(f"Warning: Error calculating correlation for feature {i}: {e}")
                scores[i] = 0.0
        return scores
    
    def _chi_squared_test(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate Chi-squared test scores."""
        try:
            # Ensure non-negative values for chi2 test
            X_non_neg = X - X.min(axis=0) + 1e-8
            chi2_scores, _ = chi2(X_non_neg, y)
            # Handle any NaN or infinite values
            chi2_scores = np.nan_to_num(chi2_scores, nan=0.0, posinf=0.0, neginf=0.0)
            return chi2_scores
        except Exception as e:
            print(f"Warning: Error in chi-squared test: {e}")
            return np.zeros(X.shape[1])
    
    def _anova_f_test(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate ANOVA F-test scores."""
        try:
            f_scores, _ = f_classif(X, y)
            # Handle any NaN or infinite values
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            return f_scores
        except Exception as e:
            print(f"Warning: Error in ANOVA F-test: {e}")
            return np.zeros(X.shape[1])
    
    def _relief_f(self, X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
        """
        ReliefF algorithm implementation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        k : int, default=10
            Number of nearest neighbors to consider
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        
        # Normalize features
        X_norm = StandardScaler().fit_transform(X)
        
        for i in range(n_samples):
            # Current instance
            instance = X_norm[i]
            label = y[i]
            
            # Find k nearest hits (same class) and misses (different class)
            same_class_indices = np.where(y == label)[0]
            diff_class_indices = np.where(y != label)[0]
            
            # Remove current instance from same class
            same_class_indices = same_class_indices[same_class_indices != i]
            
            if len(same_class_indices) == 0 or len(diff_class_indices) == 0:
                continue
                
            # Calculate distances
            same_class_distances = np.sum((X_norm[same_class_indices] - instance) ** 2, axis=1)
            diff_class_distances = np.sum((X_norm[diff_class_indices] - instance) ** 2, axis=1)
            
            # Get k nearest neighbors
            k_hits = min(k, len(same_class_indices))
            k_misses = min(k, len(diff_class_indices))
            
            nearest_hits_idx = same_class_indices[np.argsort(same_class_distances)[:k_hits]]
            nearest_misses_idx = diff_class_indices[np.argsort(diff_class_distances)[:k_misses]]
            
            # Update weights
            for j in range(n_features):
                hit_diff = np.mean(np.abs(instance[j] - X_norm[nearest_hits_idx, j]))
                miss_diff = np.mean(np.abs(instance[j] - X_norm[nearest_misses_idx, j]))
                weights[j] += (miss_diff - hit_diff) / n_samples
                
        return np.abs(weights)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of target variable."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-8))
    
    def _information_gain(self, X_feature: np.ndarray, y: np.ndarray) -> float:
        """Calculate information gain for a feature."""
        # Calculate entropy of target
        entropy_y = self._entropy(y)
        
        # Discretize continuous feature into bins
        n_bins = min(10, len(np.unique(X_feature)))
        bins = np.histogram_bin_edges(X_feature, bins=n_bins)
        digitized = np.digitize(X_feature, bins[1:-1])
        
        # Calculate weighted entropy after split
        weighted_entropy = 0
        for bin_val in np.unique(digitized):
            mask = digitized == bin_val
            if np.sum(mask) > 0:
                subset_entropy = self._entropy(y[mask])
                weighted_entropy += (np.sum(mask) / len(y)) * subset_entropy
        
        return entropy_y - weighted_entropy
    
    def _gain_ratio(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate Gain Ratio for each feature."""
        gain_ratios = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # Information gain
            info_gain = self._information_gain(X[:, i], y)
            
            # Split information (entropy of feature)
            n_bins = min(10, len(np.unique(X[:, i])))
            bins = np.histogram_bin_edges(X[:, i], bins=n_bins)
            digitized = np.digitize(X[:, i], bins[1:-1])
            
            split_info = 0
            for bin_val in np.unique(digitized):
                prob = np.sum(digitized == bin_val) / len(digitized)
                if prob > 0:
                    split_info += -prob * np.log2(prob)
            
            # Gain ratio
            if split_info > 0:
                gain_ratios[i] = info_gain / split_info
            else:
                gain_ratios[i] = 0
                
        return gain_ratios
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit the ensemble feature selection.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        feature_names : list, optional
            Names of features
        """
        # Check data types before converting to numpy array
        # For DataFrame X, check if all columns are numeric
        if hasattr(X, 'dtypes'):  # Check if it's a DataFrame
            if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
                print("Warning: Non-numeric features detected. Converting to float.")
                X = X.astype(float)
        elif hasattr(X, 'dtype'):  # Check if it's a numpy array
            if not np.issubdtype(X.dtype, np.number):
                print("Warning: Non-numeric features detected. Converting to float.")
                X = X.astype(float)
        
        # Check target data type
        if hasattr(y, 'dtype'):
            if not np.issubdtype(y.dtype, np.number):
                print("Warning: Non-numeric target detected. Converting to int.")
                y = y.astype(int)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Data validation and cleaning
        print("Validating and cleaning data...")
        
        # Check for NaN values
        if np.any(np.isnan(X)):
            print("Warning: NaN values found in features. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.any(np.isnan(y)):
            print("Warning: NaN values found in target. Replacing with mode.")
            y_mode = np.nanmedian(y)
            y = np.nan_to_num(y, nan=y_mode)
        
        # Check for infinite values
        if np.any(np.isinf(X)):
            print("Warning: Infinite values found in features. Replacing with 0.")
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
        
        if np.any(np.isinf(y)):
            print("Warning: Infinite values found in target. Replacing with median.")
            y_median = np.nanmedian(y)
            y = np.nan_to_num(y, posinf=y_median, neginf=y_median)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.feature_names_ = feature_names
        
        print(f"Data shape after cleaning: X={X.shape}, y={y.shape}")
        print(f"Data types: X={X.dtype}, y={y.dtype}")
        
        print("Calculating feature scores for each method...")
        
        # Calculate scores for each method
        self.feature_scores_['pearson'] = self._pearson_correlation(X, y)
        self.feature_scores_['chi2'] = self._chi_squared_test(X, y)
        self.feature_scores_['anova'] = self._anova_f_test(X, y)
        self.feature_scores_['relieff'] = self._relief_f(X, y)
        self.feature_scores_['gain_ratio'] = self._gain_ratio(X, y)
        
        print("Feature scoring completed!")
        return self
    
    def select_features(self, thresholds: Dict[str, float]) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """
        Select features based on thresholds for each method.

        Parameters:
        -----------
        thresholds : dict
            Dictionary with method names as keys and threshold values as values
            If the value is the string 'log2', use log2(n_features) as the threshold.
            If the value is the string 'median', use the median of the scores as the threshold.

        Returns:
        --------
        selected_features : dict
            Dictionary with method names as keys and selected feature names as values
        medians : dict
            Dictionary with method names as keys and median of feature scores as values
        """
        self.selected_features_ = {}
        medians = {}
        n_features = len(self.feature_names_)
        for method, threshold in thresholds.items():
            if method in self.feature_scores_:
                scores = self.feature_scores_[method]
                # Support threshold as 'log2' or 'median' (string)
                if isinstance(threshold, str):
                    if threshold.lower() == 'log2':
                        threshold_value = math.log2(n_features)
                    elif threshold.lower() == 'median':
                        threshold_value = float(np.median(scores))
                    else:
                        raise ValueError(f"Unknown threshold string: {threshold}")
                else:
                    threshold_value = threshold
                # Select features above threshold
                selected_indices = np.where(scores >= threshold_value)[0]
                self.selected_features_[method] = [self.feature_names_[i] for i in selected_indices]
                medians[method] = float(np.median(scores))
                print(f"{method}: Selected {len(selected_indices)} features (threshold={threshold_value}) | Median score: {medians[method]}")
        return self.selected_features_, medians
    
    def ensemble_by_intersection(self) -> List[str]:
        """
        Create ensemble features by selecting common features across all methods.
        
        Returns:
        --------
        ensemble_features : list
            List of feature names selected by all methods
        """
        if not self.selected_features_:
            raise ValueError("No features selected. Run select_features() first.")
        
        # Find intersection of all selected features
        feature_sets = [set(features) for features in self.selected_features_.values()]
        self.ensemble_features_ = list(set.intersection(*feature_sets))
        
        print(f"Ensemble (intersection): Selected {len(self.ensemble_features_)} features")
        return self.ensemble_features_
    
    def ensemble_by_majority(self, min_methods: int = 4) -> List[str]:
        """
        Create ensemble features by selecting features that appear in at least min_methods methods.
        
        Parameters:
        -----------
        min_methods : int, default=4
            Minimum number of methods that must select a feature for it to be included
            
        Returns:
        --------
        majority_features : list
            List of feature names selected by at least min_methods methods
        """
        if not self.selected_features_:
            raise ValueError("No features selected. Run select_features() first.")
        
        # Count how many methods selected each feature
        feature_counts = {}
        for method, features in self.selected_features_.items():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Select features that appear in at least min_methods methods
        majority_features = [feature for feature, count in feature_counts.items() 
                           if count >= min_methods]
        
        print(f"Ensemble (majority ≥{min_methods} methods): Selected {len(majority_features)} features")
        return majority_features
    
    def get_feature_method_counts(self) -> Dict[str, int]:
        """
        Get the count of how many methods selected each feature.
        
        Returns:
        --------
        feature_counts : dict
            Dictionary with feature names as keys and count of methods as values
        """
        if not self.selected_features_:
            raise ValueError("No features selected. Run select_features() first.")
        
        feature_counts = {}
        for method, features in self.selected_features_.items():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        return feature_counts
    
    def plot_feature_scores(self, figsize: Tuple[int, int] = (15, 12)):
        """
        Plot feature scores for each method and final ensemble features.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for the plot
        """
        if not self.feature_scores_:
            raise ValueError("No feature scores available. Run fit() first.")
        
        n_methods = len(self.feature_scores_)
        n_features = len(self.feature_names_)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Ensemble Feature Selection Results', fontsize=16, fontweight='bold')
        
        # Plot individual method scores
        methods = list(self.feature_scores_.keys())
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, method in enumerate(methods):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            scores = self.feature_scores_[method]
            # Sort scores and feature names in descending order
            sorted_indices = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_indices]
            sorted_feature_names = [self.feature_names_[idx] for idx in sorted_indices]
            feature_indices = np.arange(len(scores))
            
            # Bar plot
            bars = ax.bar(feature_indices, sorted_scores, color=colors[i], alpha=0.7)
            
            # Highlight selected features if available
            if method in self.selected_features_:
                selected_indices = []
                for fname in self.selected_features_[method]:
                    if fname in sorted_feature_names:
                        try:
                            idx = sorted_feature_names.index(fname)
                            if idx < len(bars):  # Ensure index is within bounds
                                selected_indices.append(idx)
                        except ValueError:
                            continue
                
                for idx in selected_indices:
                    if idx < len(bars):  # Double-check bounds
                        bars[idx].set_color(colors[i])
                        bars[idx].set_alpha(1.0)
            
            ax.set_title(f'{method.upper()} Scores', fontweight='bold')
            ax.set_xlabel('Feature Name')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            
            # Set x-ticks to feature names
            ax.set_xticks(feature_indices)
            # Ensure we only show a subset of labels if there are too many features
            if len(sorted_feature_names) > 20:
                # Show every nth label to avoid overcrowding
                step = max(1, len(sorted_feature_names) // 20)
                visible_labels = [sorted_feature_names[i] if i % step == 0 else '' for i in range(len(sorted_feature_names))]
                ax.set_xticklabels(visible_labels, rotation=90, fontsize=8)
            else:
                ax.set_xticklabels(sorted_feature_names, rotation=90 if n_features > 10 else 0, fontsize=8)
        
        # Plot ensemble features
        ax_ensemble = axes[1, 2]
        if self.ensemble_features_:
            ensemble_indices = [self.feature_names_.index(fname) for fname in self.ensemble_features_]
            ensemble_scores = np.zeros(n_features)
            
            # Average scores across methods for ensemble features
            for idx in ensemble_indices:
                avg_score = np.mean([self.feature_scores_[method][idx] for method in methods])
                ensemble_scores[idx] = avg_score
            # Sort ensemble scores and feature names in descending order
            sorted_ensemble_indices = np.argsort(ensemble_scores)[::-1]
            sorted_ensemble_scores = ensemble_scores[sorted_ensemble_indices]
            sorted_ensemble_names = [self.feature_names_[idx] for idx in sorted_ensemble_indices]
            feature_indices = np.arange(len(sorted_ensemble_scores))
            
            bars = ax_ensemble.bar(feature_indices, sorted_ensemble_scores, color='gold', alpha=0.7)
            for idx in range(len(sorted_ensemble_scores)):
                if idx < len(sorted_ensemble_names) and sorted_ensemble_names[idx] in self.ensemble_features_:
                    if idx < len(bars):  # Ensure index is within bounds
                        bars[idx].set_alpha(1.0)
            
            ax_ensemble.set_title('Ensemble Features (Intersection)', fontweight='bold')
            ax_ensemble.set_xlabel('Feature Name')
            ax_ensemble.set_ylabel('Average Score')
            ax_ensemble.grid(True, alpha=0.3)
            ax_ensemble.set_xticks(feature_indices)
            # Ensure we only show a subset of labels if there are too many features
            if len(sorted_ensemble_names) > 20:
                # Show every nth label to avoid overcrowding
                step = max(1, len(sorted_ensemble_names) // 20)
                visible_labels = [sorted_ensemble_names[i] if i % step == 0 else '' for i in range(len(sorted_ensemble_names))]
                ax_ensemble.set_xticklabels(visible_labels, rotation=90, fontsize=8)
            else:
                ax_ensemble.set_xticklabels(sorted_ensemble_names, rotation=90 if n_features > 10 else 0, fontsize=8)
        else:
            ax_ensemble.text(0.5, 0.5, 'No ensemble features\nRun ensemble_by_intersection()', 
                           ha='center', va='center', transform=ax_ensemble.transAxes)
            ax_ensemble.set_title('Ensemble Features', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Create summary plot
        self._plot_feature_selection_summary()
    
    def _plot_feature_selection_summary(self):
        """Plot summary of feature selection across methods."""
        if not self.selected_features_:
            return
        
        # Create summary dataframe
        methods = list(self.selected_features_.keys())
        n_selected = [len(self.selected_features_[method]) for method in methods]
        
        if self.ensemble_features_:
            methods.append('Ensemble')
            n_selected.append(len(self.ensemble_features_))
        
        # Ensure methods and n_selected have the same length
        if len(methods) != len(n_selected):
            min_len = min(len(methods), len(n_selected))
            methods = methods[:min_len]
            n_selected = n_selected[:min_len]
        
        # Bar plot of number of selected features
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, n_selected, color=['blue', 'green', 'red', 'purple', 'orange', 'gold'][:len(methods)])
        
        # Add value labels on bars
        for bar, count in zip(bars, n_selected):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title('Number of Selected Features by Method', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Selection Method')
        plt.ylabel('Number of Selected Features')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Venn diagram-like visualization (simplified)
        self._plot_feature_overlap()
    
    def plot_majority_features(self, min_methods: int = 4, figsize: Tuple[int, int] = (15, 10)):
        """
        Create a separate plot for features that are common in at least min_methods methods.
        
        Parameters:
        -----------
        min_methods : int, default=4
            Minimum number of methods that must select a feature
        figsize : tuple
            Figure size for the plot
        """
        if not self.selected_features_:
            raise ValueError("No features selected. Run select_features() first.")
        
        if not self.feature_names_:
            raise ValueError("No feature names available. Run fit() first.")
        
        if not self.feature_scores_:
            raise ValueError("No feature scores available. Run fit() first.")
        
        # Get feature counts and majority features
        feature_counts = self.get_feature_method_counts()
        majority_features = self.ensemble_by_majority(min_methods)
        
        if not majority_features:
            print(f"No features selected by at least {min_methods} methods.")
            return
        
        # Ensure all majority features exist in feature_names_
        majority_features = [f for f in majority_features if f in self.feature_names_]
        if not majority_features:
            print("No valid majority features found.")
            return
        
        # Create figure with subplots (now 2x3 to accommodate the new plot)
        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Features Common in ≥{min_methods} Methods', fontsize=16, fontweight='bold')
        
        # 1. Bar plot of feature counts
        majority_counts = {feature: count for feature, count in feature_counts.items() 
                          if count >= min_methods}
        sorted_features = sorted(majority_counts.items(), key=lambda x: x[1], reverse=True)
        feature_names = [item[0] for item in sorted_features]
        counts = [item[1] for item in sorted_features]
        
        # Filter out features that don't exist in feature_names_
        valid_feature_names = []
        valid_counts = []
        for feature, count in zip(feature_names, counts):
            if feature in self.feature_names_:
                valid_feature_names.append(feature)
                valid_counts.append(count)
        
        feature_names = valid_feature_names
        counts = valid_counts
        
        if not feature_names:
            print("No valid features to plot.")
            return
        
        bars = ax1.bar(range(len(feature_names)), counts, 
                      color=['gold' if count == 5 else 'orange' if count == 4 else 'lightblue' 
                             for count in counts])
        ax1.set_title(f'Feature Selection Count (≥{min_methods} methods)', fontweight='bold')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Number of Methods')
        ax1.set_xticks(range(len(feature_names)))
        # Ensure we only show a subset of labels if there are too many features
        if len(feature_names) > 15:
            # Show every nth label to avoid overcrowding
            step = max(1, len(feature_names) // 15)
            visible_labels = [feature_names[i] if i % step == 0 else '' for i in range(len(feature_names))]
            ax1.set_xticklabels(visible_labels, rotation=45, ha='right')
        else:
            ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
         
        # 2. Average scores for majority features
        methods = list(self.feature_scores_.keys())
        avg_scores = []
        valid_feature_names = []
        valid_counts = []
        
        for feature in feature_names:
            if feature in self.feature_names_:
                feature_idx = self.feature_names_.index(feature)
                avg_score = np.mean([self.feature_scores_[method][feature_idx] for method in methods])
                avg_scores.append(avg_score)
                valid_feature_names.append(feature)
                # Get the corresponding count
                feature_count_idx = feature_names.index(feature)
                if feature_count_idx < len(counts):
                    valid_counts.append(counts[feature_count_idx])
        
        # Update the arrays to use only valid features
        feature_names = valid_feature_names
        counts = valid_counts
        
        # Ensure all arrays have the same length
        min_len = min(len(feature_names), len(avg_scores), len(counts))
        if min_len == 0:
            print("No valid features to plot.")
            return
        
        feature_names = feature_names[:min_len]
        avg_scores = avg_scores[:min_len]
        counts = counts[:min_len]
        
        # Sort features by average scores in descending order, prioritizing 5-method features (yellow) then 4-method features (orange)
        # Create tuples of (feature_name, avg_score, count) for sorting
        feature_data = list(zip(feature_names, avg_scores, counts))
        
        # Sort by count first (5 methods first, then 4 methods), then by average score in descending order
        feature_data.sort(key=lambda x: (-x[2], -x[1]))  # Sort by count descending, then by score descending
        
        # Unzip the sorted data
        feature_names, avg_scores, counts = zip(*feature_data)
        
        bars2 = ax2.bar(range(len(feature_names)), avg_scores, 
                        color=['gold' if count == 5 else 'orange' if count == 4 else 'lightblue' 
                               for count in counts])
         
        # 2.5. All features sorted by average scores (descending) with color coding
        # Get all features and their average scores
        all_feature_scores = []
        all_feature_names = []
        all_feature_counts = []
        
        for feature in self.feature_names_:
            if feature in feature_counts:
                feature_idx = self.feature_names_.index(feature)
                avg_score = np.mean([self.feature_scores_[method][feature_idx] for method in methods])
                all_feature_scores.append(avg_score)
                all_feature_names.append(feature)
                all_feature_counts.append(feature_counts[feature])
        
        # Sort all features by average score in descending order
        all_feature_data = list(zip(all_feature_names, all_feature_scores, all_feature_counts))
        all_feature_data.sort(key=lambda x: -x[1])  # Sort by score descending
        
        # Unzip the sorted data
        all_feature_names, all_feature_scores, all_feature_counts = zip(*all_feature_data)
        
        bars5 = ax5.bar(range(len(all_feature_names)), all_feature_scores, 
                        color=['gold' if count == 5 else 'orange' if count == 4 else 'lightblue' 
                               for count in all_feature_counts])
        ax5.set_title('All Features Sorted by Average Score (Descending)', fontweight='bold')
        ax5.set_xlabel('Features')
        ax5.set_ylabel('Average Score')
        ax5.set_xticks(range(len(all_feature_names)))
        
        # Show only a subset of labels to avoid overcrowding
        if len(all_feature_names) > 20:
            step = max(1, len(all_feature_names) // 20)
            visible_labels = [all_feature_names[i] if i % step == 0 else '' for i in range(len(all_feature_names))]
            ax5.set_xticklabels(visible_labels, rotation=45, ha='right', fontsize=8)
        else:
            ax5.set_xticklabels(all_feature_names, rotation=45, ha='right', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
                # Add score labels on bars for top features
        for i, (bar, score) in enumerate(zip(bars5, all_feature_scores)):
            if i < 10:  # Only show labels for top 10 features
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gold', label='5 methods'),
            Patch(facecolor='orange', label='4 methods'),
            Patch(facecolor='lightblue', label='3 methods')
        ]
        ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 2.6. Summary table of top features by average score
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create table data for top 15 features
        table_data = []
        headers = ['Rank', 'Feature', 'Avg Score', 'Methods']
        
        for i, (feature, score, count) in enumerate(zip(all_feature_names[:15], all_feature_scores[:15], all_feature_counts[:15])):
            table_data.append([i+1, feature, f'{score:.4f}', count])
        
        # Create table
        table = ax6.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         colWidths=[0.1, 0.4, 0.2, 0.1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows based on method count
        for i in range(1, len(table_data) + 1):
            count = table_data[i-1][3]
            if count == 5:
                color = '#FFD700'  # Gold
            elif count == 4:
                color = '#FFA500'  # Orange
            else:
                color = '#ADD8E6'  # Light blue
            
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
        
        ax6.set_title('Top 15 Features by Average Score', fontweight='bold', pad=20)
        ax2.set_title('Average Feature Scores', fontweight='bold')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Average Score')
        ax2.set_xticks(range(len(feature_names)))
        # Ensure we only show a subset of labels if there are too many features
        if len(feature_names) > 15:
            # Show every nth label to avoid overcrowding
            step = max(1, len(feature_names) // 15)
            visible_labels = [feature_names[i] if i % step == 0 else '' for i in range(len(feature_names))]
            ax2.set_xticklabels(visible_labels, rotation=45, ha='right')
        else:
            ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Method-wise breakdown for majority features
        method_data = []
        method_names = []
        
        for method in methods:
            method_scores = []
            for feature in feature_names:
                if feature in self.feature_names_:
                    feature_idx = self.feature_names_.index(feature)
                    method_scores.append(self.feature_scores_[method][feature_idx])
            method_data.append(method_scores)
            method_names.append(method.upper())
        
        # Ensure all method_data arrays have the same length
        if method_data:
            min_length = min(len(scores) for scores in method_data)
            if min_length > 0:
                method_data = [scores[:min_length] for scores in method_data]
                feature_names = feature_names[:min_length]
            else:
                method_data = []
                feature_names = []
        
        # Create heatmap
        if method_data and len(method_data) > 0:
            method_data = np.array(method_data).T  # Transpose for correct orientation
            im = ax3.imshow(method_data, cmap='YlOrRd', aspect='auto')
            ax3.set_title('Feature Scores by Method', fontweight='bold')
            ax3.set_xlabel('Methods')
            ax3.set_ylabel('Features')
            ax3.set_xticks(range(len(method_names)))
            ax3.set_yticks(range(len(feature_names)))
            ax3.set_xticklabels(method_names)
            ax3.set_yticklabels(feature_names)
            
            # Add text annotations
            for i in range(len(feature_names)):
                for j in range(len(method_names)):
                    if i < method_data.shape[0] and j < method_data.shape[1]:
                        text = ax3.text(j, i, f'{method_data[i, j]:.3f}',
                                       ha="center", va="center", color="black", fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Scores by Method', fontweight='bold')
        
        # 4. Pie chart showing distribution of selection counts
        count_distribution = {}
        for count in counts:
            count_distribution[count] = count_distribution.get(count, 0) + 1
        
        if count_distribution and len(count_distribution) > 0:
            labels = [f'{count} methods' for count in sorted(count_distribution.keys())]
            sizes = list(count_distribution.values())
            colors = ['gold', 'orange', 'lightblue'][:len(sizes)]
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, 
                                               autopct='%1.1f%%', startangle=90)
            ax4.set_title('Distribution of Method Selection Counts', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Distribution of Method Selection Counts', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed information
        print(f"\n{'='*60}")
        print(f"FEATURES COMMON IN ≥{min_methods} METHODS")
        print(f"{'='*60}")
        print(f"Total features selected by ≥{min_methods} methods: {len(majority_features)}")
        print(f"Features selected by all 5 methods: {len([f for f, c in majority_counts.items() if c == 5])}")
        print(f"Features selected by exactly 4 methods: {len([f for f, c in majority_counts.items() if c == 4])}")
        
        print(f"\nDetailed breakdown:")
        # Sort features by count first (5 methods first, then 4 methods), then by average score in descending order
        detailed_features = []
        for feature, count in sorted_features:
            if feature in self.feature_names_:
                feature_idx = self.feature_names_.index(feature)
                avg_score = np.mean([self.feature_scores_[method][feature_idx] for method in methods])
                detailed_features.append((feature, count, avg_score))
        
        # Sort by count descending, then by average score descending
        detailed_features.sort(key=lambda x: (-x[1], -x[2]))
        
        for feature, count, avg_score in detailed_features:
            print(f"  {feature}: {count} methods, avg_score={avg_score:.4f}")
        
        return majority_features, feature_counts
    
    def _plot_feature_overlap(self):
        """Plot feature overlap between methods."""
        if not self.selected_features_ or len(self.selected_features_) < 2:
            return
        
        # Create overlap matrix
        methods = list(self.selected_features_.keys())
        n_methods = len(methods)
        overlap_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i, j] = len(self.selected_features_[method1])
                else:
                    overlap = len(set(self.selected_features_[method1]) & 
                                set(self.selected_features_[method2]))
                    overlap_matrix[i, j] = overlap
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(overlap_matrix, annot=True, fmt='.0f', 
                   xticklabels=[m.upper() for m in methods],
                   yticklabels=[m.upper() for m in methods],
                   cmap='Blues', cbar_kws={'label': 'Number of Common Features'})
        plt.title('Feature Overlap Between Methods', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def get_feature_ranking(self) -> pd.DataFrame:
        """
        Get ranking of features across all methods.
        
        Returns:
        --------
        ranking_df : pd.DataFrame
            DataFrame with feature rankings for each method
        """
        if not self.feature_scores_:
            raise ValueError("No feature scores available. Run fit() first.")
        
        ranking_data = {'Feature': self.feature_names_}
        
        for method, scores in self.feature_scores_.items():
            # Convert scores to rankings (1 = best)
            rankings = np.argsort(np.argsort(scores)[::-1]) + 1
            ranking_data[f'{method}_rank'] = rankings
            ranking_data[f'{method}_score'] = scores
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Add ensemble ranking (average rank)
        rank_columns = [col for col in ranking_df.columns if col.endswith('_rank')]
        ranking_df['ensemble_avg_rank'] = ranking_df[rank_columns].mean(axis=1)
        
        return ranking_df.sort_values('ensemble_avg_rank')


def quick_pearson_correlation_demo():
    """
    Quick demonstration of Pearson correlation analysis for the cgan_balanced_minmax dataset.
    This function can be called independently to test the correlation functionality.
    """
    print("Quick Pearson Correlation Demo for cgan_balanced_minmax dataset")
    print("="*60)
    
    try:
        # Load dataset
        print("Loading cgan_balanced_minmax dataset...")
        cgan_balanced_minmax = pd.read_csv('../cgan_balanced_minmax.csv')
        
        # Extract features (excluding label and type columns)
        features = cgan_balanced_minmax.drop(['label', 'type'], axis=1)
        
        print(f"Dataset loaded successfully!")
        print(f"Features shape: {features.shape}")
        print(f"Feature columns: {list(features.columns)[:5]}... (showing first 5)")
        
        # Create correlation map with median threshold
        print("\nCreating correlation map with median threshold...")
        corr_matrix, filtered_corr, threshold = create_pearson_correlation_map(
            data=features,
            threshold_type='median',
            plot_title='CGAN Balanced MinMax - Median Threshold'
        )
        
        # Create correlation map with log2 threshold
        print("\nCreating correlation map with log2 threshold...")
        corr_matrix_log2, filtered_corr_log2, threshold_log2 = create_pearson_correlation_map(
            data=features,
            threshold_type='log2',
            plot_title='CGAN Balanced MinMax - Log2 Threshold'
        )
        
        # Analyze target correlations
        print("\nAnalyzing feature correlations with target variable...")
        target_correlations = analyze_feature_correlations(
            data=cgan_balanced_minmax,
            target_col='label',
            top_n=20
        )
        
        print("\nQuick demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during quick demo: {str(e)}")
        return False


def demonstrate_pearson_correlation_analysis():
    """
    Demonstrate the new Pearson correlation analysis functionality with cgan_balanced_minmax dataset.
    """
    print("\n" + "="*60)
    print("PEARSON CORRELATION ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Load the cgan_balanced_minmax dataset
    print("Loading cgan_balanced_minmax dataset...")
    cgan_balanced_minmax = pd.read_csv('../cgan_balanced_minmax.csv')
    
    print(f"Dataset shape: {cgan_balanced_minmax.shape}")
    print(f"Columns: {list(cgan_balanced_minmax.columns)}")
    
    # Separate features and target
    features = cgan_balanced_minmax.drop(['label', 'type'], axis=1)
    target = cgan_balanced_minmax['label']
    
    print(f"Features shape: {features.shape}")
    print(f"Target unique values: {target.unique()}")
    
    # 1. Create correlation map with median threshold
    print("\n1. Creating correlation map with median threshold...")
    corr_matrix_median, filtered_corr_median, threshold_median = create_pearson_correlation_map(
        data=features,
        threshold_type='median',
        plot_title='CGAN Balanced MinMax Dataset - Median Threshold'
    )
    
    # 2. Create correlation map with log2 threshold
    print("\n2. Creating correlation map with log2 threshold...")
    corr_matrix_log2, filtered_corr_log2, threshold_log2 = create_pearson_correlation_map(
        data=features,
        threshold_type='log2',
        plot_title='CGAN Balanced MinMax Dataset - Log2 Threshold'
    )
    
    # 3. Create correlation map with custom threshold (0.5)
    print("\n3. Creating correlation map with custom threshold (0.5)...")
    corr_matrix_custom, filtered_corr_custom, threshold_custom = create_pearson_correlation_map(
        data=features,
        threshold_type='custom',
        custom_threshold=0.5,
        plot_title='CGAN Balanced MinMax Dataset - Custom Threshold (0.5)'
    )
    
    # 4. Create correlation map with no threshold (show all correlations)
    print("\n4. Creating correlation map with no threshold...")
    corr_matrix_none, filtered_corr_none, threshold_none = create_pearson_correlation_map(
        data=features,
        threshold_type='none',
        plot_title='CGAN Balanced MinMax Dataset - All Correlations'
    )
    
    # 5. Analyze feature correlations with target
    print("\n5. Analyzing feature correlations with target variable...")
    target_correlations = analyze_feature_correlations(
        data=cgan_balanced_minmax,
        target_col='label',
        top_n=25
    )
    
    # 6. Compare different threshold strategies
    print("\n6. Comparing different threshold strategies...")
    threshold_comparison = pd.DataFrame({
        'Threshold Type': ['Median', 'Log2', 'Custom (0.5)', 'None'],
        'Threshold Value': [threshold_median, threshold_log2, threshold_custom, threshold_none],
        'Non-zero Correlations': [
            np.count_nonzero(filtered_corr_median.values),
            np.count_nonzero(filtered_corr_log2.values),
            np.count_nonzero(filtered_corr_custom.values),
            np.count_nonzero(filtered_corr_none.values)
        ],
        'Total Correlations': [
            filtered_corr_median.values.size,
            filtered_corr_log2.values.size,
            filtered_corr_log2.values.size,
            filtered_corr_none.values.size
        ]
    })
    
    threshold_comparison['Percentage'] = (threshold_comparison['Non-zero Correlations'] / 
                                        threshold_comparison['Total Correlations'] * 100).round(1)
    
    print("\nThreshold Strategy Comparison:")
    print(threshold_comparison.to_string(index=False))
    
    # Save comparison to CSV
    threshold_comparison.to_csv('threshold_comparison.csv', index=False)
    print("\nThreshold comparison saved to 'threshold_comparison.csv'")
    
    # 7. Create a comprehensive correlation summary
    print("\n7. Creating comprehensive correlation summary...")
    
    # Get feature names
    feature_names = features.columns.tolist()
    
    # Create summary DataFrame
    correlation_summary = []
    
    for i, feature1 in enumerate(feature_names):
        for j, feature2 in enumerate(feature_names):
            if i < j:  # Only upper triangle to avoid duplicates
                corr_val = corr_matrix_median.loc[feature1, feature2]
                abs_corr = abs(corr_val)
                
                # Determine which thresholds this correlation passes
                passes_median = abs_corr >= threshold_median
                passes_log2 = abs_corr >= threshold_log2
                passes_custom = abs_corr >= threshold_custom
                
                correlation_summary.append({
                    'Feature1': feature1,
                    'Feature2': feature2,
                    'Correlation': corr_val,
                    'Abs_Correlation': abs_corr,
                    'Passes_Median': passes_median,
                    'Passes_Log2': passes_log2,
                    'Passes_Custom': passes_custom
                })
    
    correlation_summary_df = pd.DataFrame(correlation_summary)
    correlation_summary_df = correlation_summary_df.sort_values('Abs_Correlation', ascending=False)
    
    # Save comprehensive summary
    correlation_summary_df.to_csv('correlation_summary_comprehensive.csv', index=False)
    print("Comprehensive correlation summary saved to 'correlation_summary_comprehensive.csv'")
    
    # Show top correlations
    print(f"\nTop 20 feature correlations:")
    print(correlation_summary_df[['Feature1', 'Feature2', 'Correlation', 'Abs_Correlation']].head(20).to_string(index=False))
    
    print("\n" + "="*60)
    print("PEARSON CORRELATION ANALYSIS COMPLETED!")
    print("="*60)
    
    return {
        'correlation_matrices': {
            'median': corr_matrix_median,
            'log2': corr_matrix_log2,
            'custom': corr_matrix_custom,
            'none': corr_matrix_none
        },
        'filtered_matrices': {
            'median': filtered_corr_median,
            'log2': filtered_corr_log2,
            'custom': filtered_corr_custom,
            'none': filtered_corr_none
        },
        'thresholds': {
            'median': threshold_median,
            'log2': threshold_log2,
            'custom': threshold_custom,
            'none': threshold_none
        },
        'target_correlations': target_correlations,
        'threshold_comparison': threshold_comparison,
        'correlation_summary': correlation_summary_df
    }


def example_pearson_correlation_usage():
    """
    Simple examples showing how to use the Pearson correlation functions.
    """
    print("\n" + "="*60)
    print("PEARSON CORRELATION USAGE EXAMPLES")
    print("="*60)
    
    print("""
    # Example 1: Load dataset and create correlation map with median threshold
    cgan_balanced_minmax = pd.read_csv('path/to/cgan_balanced_minmax.csv')
    features = cgan_balanced_minmax.drop(['label', 'type'], axis=1)
    
    # Create correlation map with median threshold
    corr_matrix, filtered_corr, threshold = create_pearson_correlation_map(
        data=features,
        threshold_type='median',
        plot_title='CGAN Dataset - Median Threshold'
    )
    
    # Example 2: Create correlation map with log2 threshold
    corr_matrix_log2, filtered_corr_log2, threshold_log2 = create_pearson_correlation_map(
        data=features,
        threshold_type='log2',
        plot_title='CGAN Dataset - Log2 Threshold'
    )
    
    # Example 3: Create correlation map with custom threshold
    corr_matrix_custom, filtered_corr_custom, threshold_custom = create_pearson_correlation_map(
        data=features,
        threshold_type='custom',
        custom_threshold=0.5,
        plot_title='CGAN Dataset - Custom Threshold (0.5)'
    )
    
    # Example 4: Analyze feature correlations with target
    target_correlations = analyze_feature_correlations(
        data=cgan_balanced_minmax,
        target_col='label',
        top_n=20
    )
    
    # Example 5: Quick demo function
    quick_pearson_correlation_demo()
    """)
    
    print("\nThese functions provide:")
    print("- Pearson correlation coefficient calculation")
    print("- Selectable thresholds (median, log2, custom, none)")
    print("- Visual correlation maps (full and filtered)")
    print("- Target correlation analysis")
    print("- Comprehensive correlation summaries")
    print("- CSV export functionality")


#==============================================================================
#==============================================================================
#==============================================================================
# from ensemble_feature_selection import EnsembleFeatureSelection

def load_your_data():
    """
    Replace this function with your own data loading logic.
    
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target variable
    feature_names : list
        Names of features (optional)
    """
    
  
    df_res_39 = pd.read_csv('../No-GAN_No-EFS(resampled_39).csv')
    features = df_res_39.drop(['label', 'type'], axis =1)
    target = df_res_39['label']
    target.unique()
    features.shape
    
    X_train, X_test, y_train, y_test = train_test_split(
                                                features, target, stratify=target,
                                                test_size=0.2, random_state=1337
                                           )
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    X = X_train
    y = y_train
    feature_names = X.columns.tolist()
     
    return X, y, feature_names


def main():
    """Main function demonstrating ensemble feature selection usage."""
    
    
   
  
    print("Loading your data...")
    X, y, feature_names = load_your_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize ensemble feature selector
    efs = EnsembleFeatureSelection()
    
    # Fit the feature selection methods
    print("\nFitting ensemble feature selection...")
    efs.fit(X, y, feature_names)
    
          
    # Define custom thresholds for your data
    # You may need to adjust these based on your data characteristics
    
    thresholds = {
        'pearson': 'median',    # Use log2(n_features) as threshold
        'chi2': 'median',       # Use log2(n_features) as threshold
        'anova': 'median',      # Use log2(n_features) as threshold
        'relieff': 'median',    # Use log2(n_features) as threshold
        'gain_ratio': 'median'  # Use log2(n_features) as threshold
    }
    
    
    # thresholds = {
    #     'pearson': 0.15,    # Minimum correlation (0-1)
    #     'chi2': 3.0,        # Minimum chi-squared score
    #     'anova': 3.0,       # Minimum F-score  
    #     'relieff': 0.02,    # Minimum ReliefF weight
    #     'gain_ratio': 0.03  # Minimum gain ratio
    # }
    
    print("\n4. Selecting features with thresholds:")
    for method, threshold in thresholds.items():
        print(f"   {method}: {threshold}")
    
    print(f"\nUsing thresholds: {thresholds}")
    
    # Select features using thresholds
    selected_features, medians = efs.select_features(thresholds)
    print("\nMedian feature scores by method:")
    for method, median in medians.items():
        print(f"{method}: {median}")
    
    # Print results for each method
    print("\nFeature selection results:")
    for method, features in selected_features.items():
        print(f"{method.upper()}: {len(features)} features selected")
        if features:
            print(f"  Selected: {features[:3]}{'...' if len(features) > 3 else ''}")
            
    
    # Create ensemble by intersection
    ensemble_features = efs.ensemble_by_intersection()
    print(f"\nEnsemble (intersection): {len(ensemble_features)} features")
    print(f"Ensemble features: {ensemble_features}")
    
    # Create ensemble by majority (4 out of 5 methods)
    majority_features = efs.ensemble_by_majority(min_methods=4)
    print(f"\nEnsemble (majority ≥4 methods): {len(majority_features)} features")
    print(f"Majority features: {majority_features}")
    
    # Get feature method counts for detailed analysis
    feature_counts = efs.get_feature_method_counts()
    print(f"\nFeature selection counts:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count} methods")
    
    # Get detailed feature ranking
    ranking_df = efs.get_feature_ranking()
    print(f"\nTop 10 features by ensemble ranking:")
    print(ranking_df[['Feature', 'ensemble_avg_rank']].head(10).to_string(index=False))
    
    print(f"\nAll features by ensemble ranking (full DataFrame):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(ranking_df)
    
    # Save the full ranking DataFrame as a CSV file
    ranking_df.to_csv('ensemble_feature_ranking.csv', index=False)
    print("\nFull feature ranking saved to 'ensemble_feature_ranking.csv'.")

    # Plot visualization (results)
    print("\nGenerating plots...")
    efs.plot_feature_scores()
    
    # Create separate plot for majority features (4 out of 5 methods)
    print("\nGenerating majority features plot...")
    majority_features, feature_counts = efs.plot_majority_features(min_methods=4)
    
    # Save majority features to CSV
    majority_df = pd.DataFrame({
        'Feature': majority_features,
        'Method_Count': [feature_counts[f] for f in majority_features],
        'Average_Score': [np.mean([efs.feature_scores_[method][efs.feature_names_.index(f)] 
                                 for method in efs.feature_scores_.keys()]) 
                         for f in majority_features]
    })
    majority_df = majority_df.sort_values('Method_Count', ascending=False)
    majority_df.to_csv('majority_features_4_5_methods.csv', index=False)
    print("\nMajority features saved to 'majority_features_4_5_methods.csv'.")
    
    # Return the selected features for further use
    return ensemble_features, majority_features, ranking_df
    # return efs, ranking_df

# ensemble_features

def advanced_usage_example(X, y, feature_names):
    """
    Advanced usage example showing different threshold strategies.
    """
    print("\n" + "="*50)
    print("ADVANCED USAGE EXAMPLE")
    print("="*50)

    efs = EnsembleFeatureSelection()
    efs.fit(X, y, feature_names)
    
    # Strategy 1: log2(n_features) threshold for all methods
    print("\nStrategy 1: Select features with threshold=log2(n_features) per method")
    log2_thresholds = {
        'pearson': 'log2',
        'chi2': 'log2',
        'anova': 'log2',
        'relieff': 'log2',
        'gain_ratio': 'log2'
    }
    print(f"log2 thresholds: {log2_thresholds}")
    selected_log2, medians_log2 = efs.select_features(log2_thresholds)
    print("\nMedian feature scores by method (log2 thresholds):")
    for method, median in medians_log2.items():
        print(f"{method}: {median}")
    ensemble_log2 = efs.ensemble_by_intersection()
    print(f"Ensemble with log2(n_features) strategy: {len(ensemble_log2)} features")
    print(f"Features: {ensemble_log2}")
    
    # Majority features analysis for log2 strategy
    majority_log2 = efs.ensemble_by_majority(min_methods=4)
    print(f"Majority features (≥4 methods) with log2 strategy: {len(majority_log2)} features")
    print(f"Majority features: {majority_log2}")
    
    # Strategy 2: Percentile-based thresholds
    print(f"\nStrategy 2: Select features above 75th percentile")
    
    percentile_thresholds = {}
    for method, scores in efs.feature_scores_.items():
        percentile_thresholds[method] = np.percentile(scores, 75)
    
    print(f"Percentile thresholds: {percentile_thresholds}")
    
    selected_percentile, medians_percentile = efs.select_features(percentile_thresholds)
    print("\nMedian feature scores by method (percentile thresholds):")
    for method, median in medians_percentile.items():
        print(f"{method}: {median}")
    ensemble_percentile = efs.ensemble_by_intersection()
    
    print(f"Ensemble with percentile strategy: {len(ensemble_percentile)} features")
    print(f"Features: {ensemble_percentile}")
    
    # Majority features analysis for percentile strategy
    majority_percentile = efs.ensemble_by_majority(min_methods=4)
    print(f"Majority features (≥4 methods) with percentile strategy: {len(majority_percentile)} features")
    print(f"Majority features: {majority_percentile}")

def test_majority_features(X=None, y=None, feature_names=None):
    """
    Test function to demonstrate majority features analysis with sample data.
    """
    print("\n" + "="*60)
    print("TESTING MAJORITY FEATURES ANALYSIS")
    print("="*60)
    
    # Use provided data or create sample data
    if X is None or y is None or feature_names is None:
        # Create sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=20, n_informative=10, 
                                 n_redundant=5, n_clusters_per_class=1, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        print(f"Sample data created: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        print(f"Using provided data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize and fit ensemble feature selector
    efs = EnsembleFeatureSelection()
    efs.fit(X, y, feature_names)
    
    # Define thresholds
    thresholds = {
        'pearson': 'median',
        'chi2': 'median', 
        'anova': 'median',
        'relieff': 'median',
        'gain_ratio': 'median'
    }
    
    # Select features
    selected_features, medians = efs.select_features(thresholds)
    
    # Get majority features (4 out of 5 methods)
    majority_features = efs.ensemble_by_majority(min_methods=4)
    print(f"\nMajority features (≥4 methods): {len(majority_features)}")
    print(f"Features: {majority_features}")
    
    # Get feature counts
    feature_counts = efs.get_feature_method_counts()
    print(f"\nFeature selection counts:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count} methods")
    
    # Create majority features plot
    print("\nGenerating majority features plot...")
    majority_features, feature_counts = efs.plot_majority_features(min_methods=4)
    
    return majority_features, feature_counts

def debug_data_issues():
    """
    Debug function to identify data issues that might cause the TypeError.
    """
    print("\n" + "="*60)
    print("DEBUGGING DATA ISSUES")
    print("="*60)
    
    try:
        # Load your data
        X, y, feature_names = load_your_data()
        
        print(f"Data loaded: X={X.shape}, y={y.shape}")
        print(f"Feature names count: {len(feature_names)}")
        
        # Check for data issues
        print("\nChecking for data issues...")
        
        # Check for NaN values
        nan_features = np.isnan(X).sum(axis=0)
        nan_target = np.isnan(y).sum()
        
        print(f"NaN values in features: {nan_features.sum()} total, max per feature: {nan_features.max()}")
        print(f"NaN values in target: {nan_target}")
        
        # Check for infinite values
        inf_features = np.isinf(X).sum(axis=0)
        inf_target = np.isinf(y).sum()
        
        print(f"Infinite values in features: {inf_features.sum()} total, max per feature: {inf_features.max()}")
        print(f"Infinite values in target: {inf_target}")
        
        # Check data types
        if hasattr(X, 'dtypes'):  # DataFrame
            print(f"Feature data types: {X.dtypes.unique()}")
        elif hasattr(X, 'dtype'):  # Numpy array
            print(f"Feature data type: {X.dtype}")
        else:
            print(f"Feature data type: {type(X)}")
            
        if hasattr(y, 'dtype'):
            print(f"Target data type: {y.dtype}")
        else:
            print(f"Target data type: {type(y)}")
        
        # Check for zero variance features
        zero_var_features = np.var(X, axis=0) == 0
        print(f"Zero variance features: {zero_var_features.sum()}")
        
        # Check target distribution
        unique_targets, target_counts = np.unique(y, return_counts=True)
        print(f"Target distribution: {dict(zip(unique_targets, target_counts))}")
        
        # Test individual methods
        print("\nTesting individual feature selection methods...")
        
        efs = EnsembleFeatureSelection()
        
        # Test each method individually
        print("Testing Pearson correlation...")
        pearson_scores = efs._pearson_correlation(X.to_numpy(), y.to_numpy())
        print(f"Pearson scores shape: {pearson_scores.shape}")
        print(f"Pearson scores range: {pearson_scores.min():.4f} to {pearson_scores.max():.4f}")
        
        print("Testing Chi-squared test...")
        chi2_scores = efs._chi_squared_test(X.to_numpy(), y.to_numpy())
        print(f"Chi2 scores shape: {chi2_scores.shape}")
        print(f"Chi2 scores range: {chi2_scores.min():.4f} to {chi2_scores.max():.4f}")
        
        print("Testing ANOVA F-test...")
        anova_scores = efs._anova_f_test(X.to_numpy(), y.to_numpy())
        print(f"ANOVA scores shape: {anova_scores.shape}")
        print(f"ANOVA scores range: {anova_scores.min():.4f} to {anova_scores.max():.4f}")
        
        print("All individual tests passed!")
        return True
        
    except Exception as e:
        print(f"Error during debugging: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


###############################################################################
###############################################################################
###############################################################################
if __name__ == "__main__":
    # First, debug any data issues
    print("\n" + "="*50)
    print("DEBUGGING DATA ISSUES FIRST...")
    print("="*50)
    debug_success = debug_data_issues()
    
    if debug_success:
        print("\nData debugging completed successfully. Proceeding with main analysis...")
        
        # Run basic example
        ensemble_features, majority_features, rankings = main()
        
        # Get the data variables from the main function
        X, y, feature_names = load_your_data()
        
        # Run advanced example
        advanced_usage_example(X, y, feature_names)
        
        # Run test example
        test_majority_features(X, y, feature_names)
        
        # Run Pearson correlation analysis demonstration
        print("\n" + "="*50)
        print("RUNNING PEARSON CORRELATION ANALYSIS...")
        print("="*50)
        
        # Run the full Pearson correlation analysis demonstration
        correlation_results = demonstrate_pearson_correlation_analysis()
        
        print("\n" + "="*50)
        print("USAGE EXAMPLE COMPLETED!")
        print("="*50)
        
        # Show usage examples
        example_pearson_correlation_usage()
    else:
        print("\nData debugging failed. Please check your data and try again.")


#==============================================================================
# INDEPENDENT FUNCTION CALLS
#==============================================================================

# To run only the Pearson correlation analysis independently:
# quick_pearson_correlation_demo()

# To run the full demonstration independently:
# demonstrate_pearson_correlation_analysis()

# To see usage examples:
# example_pearson_correlation_usage()

# To test the fixes independently:
def test_fixes():
    """
    Test function to verify that the TypeError fixes work.
    """
    print("Testing the TypeError fixes...")
    
    try:
        # Test with simple data first
        X, y, feature_names = load_your_data()
        
        # Initialize ensemble feature selector
        efs = EnsembleFeatureSelection()
        
        # This should not raise the TypeError anymore
        print("Fitting ensemble feature selection...")
        efs.fit(X, y, feature_names)
        
        print("✓ Fit completed successfully!")
        
        # Test feature selection
        thresholds = {
            'pearson': 'median',
            'chi2': 'median',
            'anova': 'median',
            'relieff': 'median',
            'gain_ratio': 'median'
        }
        
        print("Selecting features...")
        selected_features, medians = efs.select_features(thresholds)
        
        print("✓ Feature selection completed successfully!")
        print(f"Selected features by method:")
        for method, features in selected_features.items():
            print(f"  {method}: {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Uncomment the line below to test the fixes:
# test_fixes()

# Quick test function to verify the fixes
def quick_test():
    """
    Quick test to verify the AttributeError fixes work.
    """
    print("Running quick test to verify fixes...")
    
    try:
        # Test with your data
        X, y, feature_names = load_your_data()
        
        print(f"✓ Data loaded successfully: X={X.shape}, y={y.shape}")
        print(f"✓ Feature names count: {len(feature_names)}")
        
        # Test data type checking (this was causing the AttributeError)
        if hasattr(X, 'dtypes'):  # DataFrame
            print(f"✓ Feature data types: {X.dtypes.unique()}")
        elif hasattr(X, 'dtype'):  # Numpy array
            print(f"✓ Feature data type: {X.dtype}")
        else:
            print(f"✓ Feature data type: {type(X)}")
            
        if hasattr(y, 'dtype'):
            print(f"✓ Target data type: {y.dtype}")
        else:
            print(f"✓ Target data type: {type(y)}")
        
        # Test if we can create the ensemble feature selector
        efs = EnsembleFeatureSelection()
        print("✓ EnsembleFeatureSelection created successfully")
        
        print("\n✓ All tests passed! The AttributeError has been fixed.")
        return True
        
    except Exception as e:
        print(f"✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Uncomment the line below to run the quick test:
# quick_test()

# Simple test function to verify all fixes work
def run_simple_test():
    """
    Simple test function to verify that all the fixes work correctly.
    """
    print("Running simple test to verify all fixes...")
    
    try:
        # Test data loading
        X, y, feature_names = load_your_data()
        print(f"✓ Data loaded successfully: X={X.shape}, y={y.shape}")
        
        # Test ensemble feature selection
        efs = EnsembleFeatureSelection()
        print("✓ EnsembleFeatureSelection created successfully")
        
        # Test fitting
        efs.fit(X, y, feature_names)
        print("✓ Fit completed successfully")
        
        # Test feature selection
        thresholds = {
            'pearson': 'median',
            'chi2': 'median',
            'anova': 'median',
            'relieff': 'median',
            'gain_ratio': 'median'
        }
        
        selected_features, medians = efs.select_features(thresholds)
        print("✓ Feature selection completed successfully")
        
        # Test ensemble methods
        ensemble_features = efs.ensemble_by_intersection()
        majority_features = efs.ensemble_by_majority(min_methods=4)
        print("✓ Ensemble methods completed successfully")
        
        print(f"\n✓ All tests passed! The code is working correctly.")
        print(f"  - Selected features by method: {len(selected_features)}")
        print(f"  - Ensemble features: {len(ensemble_features)}")
        print(f"  - Majority features: {len(majority_features)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Uncomment the line below to run the simple test:
# run_simple_test()

# Test function to verify the AttributeError fixes work
def test_attribute_error_fixes():
    """
    Test function to verify that the AttributeError fixes work correctly.
    """
    print("Testing the AttributeError fixes...")
    
    try:
        # Test data loading
        X, y, feature_names = load_your_data()
        print(f"✓ Data loaded successfully: X={X.shape}, y={y.shape}")
        
        # Test data type checking (this was causing the AttributeError)
        if hasattr(X, 'dtypes'):  # DataFrame
            print(f"✓ Feature data types: {X.dtypes.unique()}")
        elif hasattr(X, 'dtype'):  # Numpy array
            print(f"✓ Feature data type: {X.dtype}")
        else:
            print(f"✓ Feature data type: {type(X)}")
            
        if hasattr(y, 'dtype'):
            print(f"✓ Target data type: {y.dtype}")
        else:
            print(f"✓ Target data type: {type(y)}")
        
        # Test ensemble feature selection
        efs = EnsembleFeatureSelection()
        print("✓ EnsembleFeatureSelection created successfully")
        
        # Test fitting (this was causing the AttributeError)
        efs.fit(X, y, feature_names)
        print("✓ Fit completed successfully")
        
        print(f"\n✓ All tests passed! The AttributeError has been fixed.")
        return True
        
    except Exception as e:
        print(f"✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Uncomment the line below to test the AttributeError fixes:
# test_attribute_error_fixes() 

###############################################################################
df_res_39 = pd.read_csv('../No-GAN_No-EFS(resampled_39).csv')
df_res_fs_14 = df_res_39[[
    'sphone_signal', 'latitude', 'longitude', 'service', 'conn_state', 'src_pkts', 
    'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes', 'dns_qclass', 'dns_qtype', 'dns_AA', 
    'dns_RD', 'dns_rejected', 'label', 'type'
    ]]

# df_res_fs_14.shape   # (3174, 14)
# df_res_fs_14.dtypes
df_res_fs_14.to_csv('../No-GAN_Yes-EFS(resampled_39_efs_14).csv', index=False)

df_gan_39 = pd.read_csv('../Yes-GAN_No-EFS(cgan_balanced_39).csv')
df_gan_fs_12 = df_gan_39[[
    'humidity', 'dns_rejected', 'dns_qclass', 'src_ip_bytes', 'dst_pkts', 'dns_AA', 
    'latitude', 'thermostat_status', 'dns_RD', 'src_pkts', 'pressure', 'missed_bytes', 
    'label', 'type'
    ]]
df_gan_fs_12.to_csv('../Yes-GAN_Yes-EFS(cgan_balanced_39_efs_12).csv', index=False)




