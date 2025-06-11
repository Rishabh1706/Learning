"""
Classical ML Milestone Project - Comprehensive Machine Learning Pipeline

This file serves as a starter template for your project.
Implement the required functionality to demonstrate your mastery of 
classical machine learning techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class MathFoundations:
    """Mathematical foundations for machine learning"""
    
    @staticmethod
    def gradient_descent(X, y, learning_rate=0.01, iterations=1000, tolerance=1e-6):
        """
        Implementation of gradient descent for linear regression
        
        Args:
            X: Features matrix
            y: Target vector
            learning_rate: Learning rate for gradient descent
            iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            weights: Trained weights
            costs: Cost history
        """
        m, n = X.shape
        weights = np.zeros(n)
        costs = []
        
        for i in range(iterations):
            # Compute predictions
            predictions = np.dot(X, weights)
            
            # Compute error
            error = predictions - y
            
            # Compute cost (MSE)
            cost = np.sum(error ** 2) / (2 * m)
            costs.append(cost)
            
            # Compute gradient
            gradient = np.dot(X.T, error) / m
            
            # Update weights
            weights_new = weights - learning_rate * gradient
            
            # Check for convergence
            if np.sum(abs(weights_new - weights)) < tolerance:
                weights = weights_new
                break
                
            weights = weights_new
            
        return weights, costs
    
    @staticmethod
    def linear_algebra_operations():
        """Demonstrate basic linear algebra operations"""
        # Vector operations
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        
        # Vector addition
        v_add = v1 + v2
        
        # Dot product
        dot_product = np.dot(v1, v2)
        
        # Matrix operations
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # Matrix multiplication
        C = np.dot(A, B)
        
        # Matrix transpose
        A_T = A.T
        
        # Matrix inverse
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = "Singular matrix, inverse doesn't exist"
            
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        return {
            "vector_addition": v_add,
            "dot_product": dot_product,
            "matrix_multiplication": C,
            "matrix_transpose": A_T,
            "matrix_inverse": A_inv,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors
        }
    
    @staticmethod
    def bayesian_probability(prior, likelihood, evidence):
        """
        Compute posterior probability using Bayes theorem
        
        P(A|B) = P(B|A) * P(A) / P(B)
        
        Args:
            prior: P(A) - Prior probability
            likelihood: P(B|A) - Likelihood
            evidence: P(B) - Evidence
            
        Returns:
            posterior: P(A|B) - Posterior probability
        """
        posterior = (likelihood * prior) / evidence
        return posterior
    
    @staticmethod
    def calculate_entropy(probabilities):
        """
        Calculate entropy from probability distribution
        
        Args:
            probabilities: List of probabilities
            
        Returns:
            entropy: Entropy value
        """
        # Ensure probabilities sum to 1
        probabilities = np.array(probabilities) / sum(probabilities)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    @staticmethod
    def calculate_kl_divergence(p, q):
        """
        Calculate Kullback-Leibler divergence between two distributions
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            kl_divergence: KL divergence value
        """
        # Ensure distributions sum to 1
        p = np.array(p) / sum(p)
        q = np.array(q) / sum(q)
        
        # Calculate KL divergence
        kl_divergence = np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))
        return kl_divergence


class DataPreparation:
    """Data loading, cleaning, and preprocessing"""
    
    @staticmethod
    def load_data(filepath):
        """Load data from file"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        return df
    
    @staticmethod
    def explore_data(df):
        """Generate descriptive statistics and basic exploratory analysis"""
        # Basic info and statistics
        info = {
            "shape": df.shape,
            "dtypes": df.dtypes,
            "missing_values": df.isnull().sum(),
            "duplicates": df.duplicated().sum(),
            "numerical_stats": df.describe(),
            "categorical_stats": df.describe(include=['object', 'category'])
        }
        
        return info
    
    @staticmethod
    def clean_data(df):
        """Clean data by handling missing values, outliers, etc."""
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                # Fill numerical missing values with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                # Fill categorical missing values with mode
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle outliers (using IQR method for numerical columns)
        for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean
    
    @staticmethod
    def hypothesis_testing(df, column1, column2=None, test_type='t-test'):
        """
        Perform hypothesis testing on column(s)
        
        Args:
            df: DataFrame
            column1: First column name
            column2: Second column name (for two-sample tests)
            test_type: Type of test ('t-test', 'chi-square', etc.)
            
        Returns:
            result: Dictionary with test results
        """
        from scipy import stats
        
        result = {}
        
        if test_type == 't-test':
            if column2:
                # Two-sample t-test
                stat, pvalue = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
                result = {
                    "test": "Two-sample t-test",
                    "statistic": stat,
                    "p_value": pvalue,
                    "significant": pvalue < 0.05
                }
            else:
                # One-sample t-test against mean=0
                stat, pvalue = stats.ttest_1samp(df[column1].dropna(), 0)
                result = {
                    "test": "One-sample t-test",
                    "statistic": stat,
                    "p_value": pvalue,
                    "significant": pvalue < 0.05
                }
        
        elif test_type == 'chi-square':
            # Create contingency table
            contingency = pd.crosstab(df[column1], df[column2] if column2 else df[column1])
            stat, pvalue, dof, expected = stats.chi2_contingency(contingency)
            result = {
                "test": "Chi-square test",
                "statistic": stat,
                "p_value": pvalue,
                "dof": dof,
                "significant": pvalue < 0.05
            }
        
        return result
    
    @staticmethod
    def visualize_data(df):
        """Create visualizations for data exploration"""
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Create distribution plots for numerical columns
        for col in numeric_cols[:min(5, len(numeric_cols))]:  # Limit to 5 columns
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[col])
            plt.title(f'Boxplot of {col}')
            
            plt.tight_layout()
            plt.show()
        
        # Create correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix')
            plt.show()
        
        # Create pairplot for sample of variables
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols[:min(4, len(numeric_cols))]])
            plt.title('Pairplot of Numeric Variables')
            plt.show()


class FeatureEngineering:
    """Feature creation, selection, and scaling"""
    
    @staticmethod
    def create_polynomial_features(X, degree=2):
        """Create polynomial features"""
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        return X_poly, poly
    
    @staticmethod
    def create_interaction_features(X):
        """Create interaction features between variables"""
        n_features = X.shape[1]
        X_interaction = X.copy()
        
        # Create all possible interactions
        for i in range(n_features):
            for j in range(i+1, n_features):
                X_interaction = np.column_stack((X_interaction, X[:, i] * X[:, j]))
        
        return X_interaction
    
    @staticmethod
    def standardize_features(X_train, X_test=None):
        """Standardize features (zero mean, unit variance)"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, scaler
    
    @staticmethod
    def select_features(X, y, method='correlation', threshold=0.1):
        """
        Select features based on specified method
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('correlation', 'mutual_info', 'rfe')
            threshold: Threshold for selection
            
        Returns:
            selected_features: Indices of selected features
        """
        if method == 'correlation':
            # Calculate correlation with target
            correlations = np.abs(np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]))
            selected_features = np.where(correlations > threshold)[0]
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            if len(np.unique(y)) < 5:  # Classification task
                mi = mutual_info_classif(X, y)
            else:  # Regression task
                mi = mutual_info_regression(X, y)
            selected_features = np.where(mi > threshold)[0]
        
        elif method == 'rfe':
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LinearRegression
            
            estimator = LinearRegression()
            selector = RFE(estimator, n_features_to_select=int(X.shape[1] * threshold))
            selector.fit(X, y)
            selected_features = np.where(selector.support_)[0]
        
        else:
            raise ValueError("Unsupported feature selection method")
        
        return selected_features


class SupervisedLearning:
    """Implementation of supervised learning models"""
    
    class CustomLogisticRegression:
        """Custom implementation of Logistic Regression"""
        
        def __init__(self, learning_rate=0.01, iterations=1000):
            self.learning_rate = learning_rate
            self.iterations = iterations
            self.weights = None
            self.bias = None
        
        def sigmoid(self, z):
            """Sigmoid activation function"""
            return 1 / (1 + np.exp(-z))
        
        def fit(self, X, y):
            """Train the model"""
            # Initialize parameters
            m, n = X.shape
            self.weights = np.zeros(n)
            self.bias = 0
            
            # Gradient descent
            for i in range(self.iterations):
                # Forward pass
                z = np.dot(X, self.weights) + self.bias
                predictions = self.sigmoid(z)
                
                # Compute gradients
                dw = (1/m) * np.dot(X.T, (predictions - y))
                db = (1/m) * np.sum(predictions - y)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            return self
        
        def predict_proba(self, X):
            """Predict probabilities"""
            z = np.dot(X, self.weights) + self.bias
            return self.sigmoid(z)
        
        def predict(self, X, threshold=0.5):
            """Predict classes based on probability threshold"""
            return (self.predict_proba(X) >= threshold).astype(int)
    
    @staticmethod
    def train_regression_models(X_train, y_train, X_test, y_test):
        """Train and evaluate regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Evaluate model
            from sklearn.metrics import mean_squared_error, r2_score
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'coefficients': model.coef_
            }
        
        return results
    
    @staticmethod
    def train_classification_models(X_train, y_train, X_test, y_test):
        """Train and evaluate classification models"""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Get probabilities (if available)
            try:
                test_proba = model.predict_proba(X_test)[:, 1]
            except:
                test_proba = None
            
            # Evaluate model
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            test_precision = precision_score(y_test, test_pred, average='weighted')
            test_recall = recall_score(y_test, test_pred, average='weighted')
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            # Calculate ROC AUC if probabilities are available
            test_roc_auc = roc_auc_score(y_test, test_proba) if test_proba is not None else None
            
            # Generate confusion matrix
            test_cm = confusion_matrix(y_test, test_pred)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'confusion_matrix': test_cm
            }
        
        return results
    
    @staticmethod
    def create_voting_classifier(models, X_train, y_train, voting='hard'):
        """Create voting classifier from multiple models"""
        # Extract model name and model from results
        estimators = [(name, model['model']) for name, model in models.items()]
        
        # Create voting classifier
        voting_clf = VotingClassifier(estimators=estimators, voting=voting)
        
        # Train voting classifier
        voting_clf.fit(X_train, y_train)
        
        return voting_clf


class UnsupervisedLearning:
    """Implementation of unsupervised learning models"""
    
    @staticmethod
    def kmeans_clustering(X, n_clusters=3):
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        return {
            'model': kmeans,
            'clusters': clusters,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_  # Sum of squared distances to closest centroid
        }
    
    @staticmethod
    def determine_optimal_clusters(X, max_clusters=10):
        """Determine optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        
        from sklearn.metrics import silhouette_score
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, clusters))
        
        # Plot elbow method
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), inertias, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    @staticmethod
    def dbscan_clustering(X, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)
        
        # Calculate number of clusters
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        return {
            'model': dbscan,
            'clusters': clusters,
            'n_clusters': n_clusters,
            'n_noise': list(clusters).count(-1)  # Number of noise points
        }
    
    @staticmethod
    def hierarchical_clustering(X, n_clusters=3, linkage='ward'):
        """Perform hierarchical clustering"""
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clusters = hc.fit_predict(X)
        
        return {
            'model': hc,
            'clusters': clusters,
            'n_clusters': n_clusters
        }
    
    @staticmethod
    def pca_dimension_reduction(X, n_components=2):
        """Perform PCA for dimensionality reduction"""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        return {
            'model': pca,
            'transformed_data': X_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
    
    @staticmethod
    def tsne_dimension_reduction(X, n_components=2, perplexity=30):
        """Perform t-SNE for dimensionality reduction"""
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        return {
            'model': tsne,
            'transformed_data': X_tsne
        }
    
    @staticmethod
    def umap_dimension_reduction(X, n_components=2, n_neighbors=15, min_dist=0.1):
        """Perform UMAP for dimensionality reduction"""
        mapper = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        X_umap = mapper.fit_transform(X)
        
        return {
            'model': mapper,
            'transformed_data': X_umap
        }
    
    @staticmethod
    def visualize_clusters(X_reduced, clusters, title='Clustering Results'):
        """Visualize clusters in 2D"""
        plt.figure(figsize=(10, 8))
        
        # Get unique clusters
        unique_clusters = np.unique(clusters)
        
        # Plot each cluster
        for cluster in unique_clusters:
            # If cluster is -1, it's noise points in DBSCAN
            if cluster == -1:
                plt.scatter(X_reduced[clusters == cluster, 0], X_reduced[clusters == cluster, 1], 
                           s=50, c='black', marker='x', label='Noise')
            else:
                plt.scatter(X_reduced[clusters == cluster, 0], X_reduced[clusters == cluster, 1], 
                           s=50, label=f'Cluster {cluster}')
        
        plt.title(title)
        plt.legend()
        plt.show()


class ModelEvaluation:
    """Model evaluation and metrics calculation"""
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_prob=None):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Add ROC AUC if probabilities are available
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """Calculate regression metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
        """Perform cross-validation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
    
    @staticmethod
    def learning_curve_analysis(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
        """Generate learning curves"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model, X=X, y=y, cv=cv, train_sizes=train_sizes)
        
        # Calculate mean and std of training and test scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.title('Learning Curves')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
        
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores
        }
    
    @staticmethod
    def plot_confusion_matrix(cm, classes=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if classes is not None:
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks + 0.5, classes)
            plt.yticks(tick_marks + 0.5, classes)
        
        plt.show()


class HyperparameterTuning:
    """Hyperparameter optimization techniques"""
    
    @staticmethod
    def grid_search_cv(model, param_grid, X, y, cv=5, scoring='accuracy'):
        """Perform grid search cross-validation"""
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
    
    @staticmethod
    def random_search_cv(model, param_distributions, X, y, cv=5, scoring='accuracy', n_iter=10):
        """Perform random search cross-validation"""
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            estimator=model, param_distributions=param_distributions, 
            n_iter=n_iter, cv=cv, scoring=scoring, random_state=42)
        
        random_search.fit(X, y)
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_model': random_search.best_estimator_,
            'cv_results': random_search.cv_results_
        }
    
    @staticmethod
    def bayesian_optimization(model, param_space, X, y, cv=5, scoring='accuracy', n_iter=10):
        """Perform Bayesian optimization using scikit-optimize"""
        from skopt import BayesSearchCV
        
        bayes_search = BayesSearchCV(
            estimator=model, search_spaces=param_space,
            n_iter=n_iter, cv=cv, scoring=scoring, random_state=42)
        
        bayes_search.fit(X, y)
        
        return {
            'best_params': bayes_search.best_params_,
            'best_score': bayes_search.best_score_,
            'best_model': bayes_search.best_estimator_,
            'cv_results': bayes_search.cv_results_
        }
    
    @staticmethod
    def plot_parameter_importance(results):
        """Plot parameter importance from optimization results"""
        from skopt.plots import plot_objective
        
        # Convert cv_results to proper format for skopt
        try:
            plot_objective(results)
            plt.show()
        except:
            print("Parameter importance plot requires scikit-optimize and GridSearchCV results in specific format")


class TimeSeriesAnalysis:
    """Implementation of time series analysis and forecasting methods"""
    
    @staticmethod
    def decompose_time_series(time_series, period=None, model='additive'):
        """
        Decompose time series into trend, seasonality and residuals
        
        Args:
            time_series: Time series data (pandas Series with DatetimeIndex)
            period: Seasonality period (e.g., 12 for monthly, 7 for weekly)
            model: Type of decomposition ('additive' or 'multiplicative')
            
        Returns:
            result: Decomposition result with trend, seasonal, and residual components
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if period is None:
            # Try to infer frequency
            if hasattr(time_series.index, 'freq') and time_series.index.freq is not None:
                # Use pandas frequency
                if time_series.index.freq.name == 'M':
                    period = 12  # Monthly data
                elif time_series.index.freq.name == 'Q':
                    period = 4   # Quarterly data
                elif time_series.index.freq.name == 'D':
                    period = 7   # Daily data (weekly seasonality)
                else:
                    period = 12  # Default
            else:
                period = 12  # Default
        
        result = seasonal_decompose(time_series, model=model, period=period)
        
        # Create plots
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        time_series.plot(ax=axes[0], title='Original')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonality')
        result.resid.plot(ax=axes[3], title='Residuals')
        
        plt.tight_layout()
        plt.show()
        
        return result
    
    @staticmethod
    def check_stationarity(time_series, window=10):
        """
        Check stationarity of a time series using ADF test and rolling statistics
        
        Args:
            time_series: Time series data
            window: Rolling window size
            
        Returns:
            result: Dictionary with test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        # Perform ADF test
        adf_result = adfuller(time_series.dropna())
        
        # Calculate rolling statistics
        rolling_mean = time_series.rolling(window=window).mean()
        rolling_std = time_series.rolling(window=window).std()
        
        # Plot rolling statistics
        plt.figure(figsize=(10, 6))
        plt.plot(time_series, color='blue', label='Original')
        plt.plot(rolling_mean, color='red', label=f'Rolling Mean ({window})')
        plt.plot(rolling_std, color='black', label=f'Rolling Std ({window})')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()
        
        # Create result dictionary
        result = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,  # p-value < 0.05 suggests stationarity
            'critical_values': adf_result[4],
            'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
        }
        
        print(f"ADF Statistic: {result['adf_statistic']:.4f}")
        print(f"p-value: {result['p_value']:.4f}")
        print(f"Is Stationary: {result['is_stationary']}")
        print("Critical Values:")
        for key, value in result['critical_values'].items():
            print(f"\t{key}: {value:.4f}")
        
        return result
    
    @staticmethod
    def make_stationary(time_series, method='differencing', lmbda=None):
        """
        Transform time series to make it stationary
        
        Args:
            time_series: Time series data
            method: Transformation method ('differencing' or 'box-cox')
            lmbda: Box-Cox transformation parameter (if None, it's estimated)
            
        Returns:
            stationary_series: Transformed stationary series
            transform_params: Parameters used for transformation
        """
        import pandas as pd
        from scipy import stats
        
        if method == 'differencing':
            # First-order differencing
            stationary_series = time_series.diff().dropna()
            transform_params = {'method': 'differencing', 'order': 1}
            
            # Plot original vs differenced
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            axes[0].plot(time_series)
            axes[0].set_title('Original Series')
            axes[1].plot(stationary_series)
            axes[1].set_title('Differenced Series')
            plt.tight_layout()
            plt.show()
            
        elif method == 'box-cox':
            # Box-Cox transformation
            data_array = time_series.values
            if lmbda is None:
                transformed_data, lmbda = stats.boxcox(data_array)
                print(f"Estimated Box-Cox parameter (lambda): {lmbda:.4f}")
            else:
                transformed_data = stats.boxcox(data_array, lmbda=lmbda)
            
            stationary_series = pd.Series(transformed_data, index=time_series.index)
            transform_params = {'method': 'box-cox', 'lambda': lmbda}
            
            # Plot original vs transformed
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            axes[0].plot(time_series)
            axes[0].set_title('Original Series')
            axes[1].plot(stationary_series)
            axes[1].set_title('Box-Cox Transformed Series')
            plt.tight_layout()
            plt.show()
        
        else:
            raise ValueError("Method must be 'differencing' or 'box-cox'")
        
        return stationary_series, transform_params
    
    @staticmethod
    def train_arima_model(time_series, order=(1,1,1), seasonal_order=None):
        """
        Train ARIMA or SARIMA model
        
        Args:
            time_series: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s) for SARIMA
            
        Returns:
            model: Fitted ARIMA/SARIMA model
            summary: Model summary
        """
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        if seasonal_order is None:
            # ARIMA model
            model = ARIMA(time_series, order=order)
            fitted_model = model.fit()
            print("ARIMA Model Summary:")
        else:
            # SARIMA model
            model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            print("SARIMA Model Summary:")
        
        print(fitted_model.summary())
        
        # Plot diagnostics
        fitted_model.plot_diagnostics(figsize=(12, 10))
        plt.tight_layout()
        plt.show()
        
        return fitted_model
    
    @staticmethod
    def auto_arima(time_series, seasonal=False, m=12):
        """
        Automatically find optimal ARIMA/SARIMA parameters
        
        Args:
            time_series: Time series data
            seasonal: Whether to include seasonal component
            m: Seasonal period
            
        Returns:
            model: Best fitted model
            order: Best ARIMA order
            seasonal_order: Best seasonal order (if applicable)
        """
        try:
            from pmdarima import auto_arima
            
            print("Finding optimal ARIMA parameters...")
            
            if seasonal:
                # With seasonal component
                model = auto_arima(
                    time_series, 
                    start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
                    start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                    m=m,  # Seasonal period
                    seasonal=True,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                print(f"Best SARIMA order: {model.order}")
                print(f"Best seasonal order: {model.seasonal_order}")
                
                return {
                    'model': model,
                    'order': model.order,
                    'seasonal_order': model.seasonal_order
                }
            else:
                # Without seasonal component
                model = auto_arima(
                    time_series,
                    start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
                    seasonal=False,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                print(f"Best ARIMA order: {model.order}")
                
                return {
                    'model': model,
                    'order': model.order,
                    'seasonal_order': None
                }
                
        except ImportError:
            print("pmdarima package not found. Install with: pip install pmdarima")
            return None
    
    @staticmethod
    def prophet_forecasting(time_series, periods=30, uncertainty_intervals=True):
        """
        Forecast using Facebook Prophet
        
        Args:
            time_series: Time series data (pandas Series with DatetimeIndex)
            periods: Number of periods to forecast
            uncertainty_intervals: Whether to include uncertainty intervals
            
        Returns:
            forecast: Forecast results
            model: Fitted Prophet model
        """
        try:
            import pandas as pd
            from prophet import Prophet
            
            # Prophet requires a specific dataframe format with 'ds' and 'y' columns
            df = pd.DataFrame({
                'ds': time_series.index,
                'y': time_series.values
            })
            
            # Create and fit model
            model = Prophet(interval_width=0.95, daily_seasonality=False)
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='D')
            
            # Make forecast
            forecast = model.predict(future)
            
            # Plot forecast
            fig = model.plot(forecast)
            plt.title('Prophet Forecast')
            plt.show()
            
            # Plot components
            fig = model.plot_components(forecast)
            plt.show()
            
            return {
                'forecast': forecast,
                'model': model
            }
            
        except ImportError:
            print("Prophet package not found. Install with: pip install prophet")
            return None
    
    @staticmethod
    def evaluate_forecast(actual, predicted, metrics=None):
        """
        Evaluate forecast performance
        
        Args:
            actual: Actual values
            predicted: Predicted values
            metrics: List of metrics to calculate (default: all)
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'mape', 'smape']
        
        results = {}
        
        if 'mse' in metrics:
            from sklearn.metrics import mean_squared_error
            results['mse'] = mean_squared_error(actual, predicted)
        
        if 'rmse' in metrics:
            from sklearn.metrics import mean_squared_error
            results['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
        
        if 'mae' in metrics:
            from sklearn.metrics import mean_absolute_error
            results['mae'] = mean_absolute_error(actual, predicted)
        
        if 'mape' in metrics:
            # Mean Absolute Percentage Error
            mask = actual != 0
            results['mape'] = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        
        if 'smape' in metrics:
            # Symmetric Mean Absolute Percentage Error
            results['smape'] = 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
        
        # Print results
        print("Forecast Evaluation Metrics:")
        for metric, value in results.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return results


class AutoML:
    """Implementation of AutoML techniques and tools"""
    
    @staticmethod
    def auto_sklearn_classification(X_train, y_train, X_test, y_test, time_left=3600, memory_limit=3072):
        """
        Automated machine learning with auto-sklearn for classification
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            time_left: Time budget in seconds
            memory_limit: Memory limit in MB
            
        Returns:
            model: Fitted auto-sklearn model
            predictions: Predictions on test data
            results: Performance metrics
        """
        try:
            import autosklearn.classification
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            print("Training Auto-sklearn classifier...")
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=time_left,
                per_run_time_limit=int(time_left/10),
                memory_limit=memory_limit,
                n_jobs=-1,
                ensemble_size=50
            )
            
            # Fit model
            automl.fit(X_train, y_train)
            
            # Print model statistics
            print(automl.sprint_statistics())
            
            # Make predictions
            y_pred = automl.predict(X_test)
            try:
                y_prob = automl.predict_proba(X_test)
            except:
                y_prob = None
            
            # Evaluate
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
            }
            
            if y_prob is not None and len(np.unique(y_test)) == 2:
                results['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
            
            # Print leaderboard
            print("\nModel Leaderboard:")
            for i, (m, w) in enumerate(automl.get_models_with_weights()):
                print(f"{i}. Weight: {w}")
                print(f"   {m}")
            
            return {
                'model': automl,
                'predictions': y_pred,
                'results': results
            }
            
        except ImportError:
            print("Auto-sklearn not installed. Install with: pip install auto-sklearn")
            return None
    
    @staticmethod
    def auto_sklearn_regression(X_train, y_train, X_test, y_test, time_left=3600, memory_limit=3072):
        """
        Automated machine learning with auto-sklearn for regression
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            time_left: Time budget in seconds
            memory_limit: Memory limit in MB
            
        Returns:
            model: Fitted auto-sklearn model
            predictions: Predictions on test data
            results: Performance metrics
        """
        try:
            import autosklearn.regression
            from sklearn.metrics import mean_squared_error, r2_score
            
            print("Training Auto-sklearn regressor...")
            automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=time_left,
                per_run_time_limit=int(time_left/10),
                memory_limit=memory_limit,
                n_jobs=-1,
                ensemble_size=50
            )
            
            # Fit model
            automl.fit(X_train, y_train)
            
            # Print model statistics
            print(automl.sprint_statistics())
            
            # Make predictions
            y_pred = automl.predict(X_test)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'rmse': rmse,
                'r2': r2
            }
            
            # Print leaderboard
            print("\nModel Leaderboard:")
            for i, (m, w) in enumerate(automl.get_models_with_weights()):
                print(f"{i}. Weight: {w}")
                print(f"   {m}")
            
            return {
                'model': automl,
                'predictions': y_pred,
                'results': results
            }
            
        except ImportError:
            print("Auto-sklearn not installed. Install with: pip install auto-sklearn")
            return None
    
    @staticmethod
    def tpot_classifier(X_train, y_train, X_test, y_test, generations=10, population_size=50, max_time_mins=60):
        """
        Automated machine learning with TPOT for classification
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            generations: Number of generations for genetic programming
            population_size: Population size for genetic programming
            max_time_mins: Maximum time in minutes
            
        Returns:
            model: Fitted TPOT model
            predictions: Predictions on test data
            results: Performance metrics
        """
        try:
            from tpot import TPOTClassifier
            from sklearn.metrics import accuracy_score, f1_score
            
            print("Training TPOT classifier...")
            tpot = TPOTClassifier(
                generations=generations,
                population_size=population_size,
                verbosity=2,
                random_state=42,
                n_jobs=-1,
                max_time_mins=max_time_mins
            )
            
            # Fit model
            tpot.fit(X_train, y_train)
            
            # Make predictions
            y_pred = tpot.predict(X_test)
            
            # Evaluate
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Export pipeline code
            print("\nBest pipeline:")
            tpot.export('tpot_exported_pipeline.py')
            
            return {
                'model': tpot,
                'predictions': y_pred,
                'results': results
            }
            
        except ImportError:
            print("TPOT not installed. Install with: pip install tpot")
            return None
    
    @staticmethod
    def h2o_automl(X_train, y_train, X_test, y_test, max_runtime_secs=3600, max_models=20, is_classification=True):
        """
        Automated machine learning with H2O AutoML
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            max_runtime_secs: Maximum runtime in seconds
            max_models: Maximum number of models to train
            is_classification: Whether it's a classification task
            
        Returns:
            model: Fitted H2O model
            predictions: Predictions on test data
            results: Performance metrics
        """
        try:
            import h2o
            from h2o.automl import H2OAutoML
            
            # Initialize H2O
            h2o.init()
            
            # Convert data to H2O frames
            train_frame = h2o.H2OFrame(pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train, columns=['target'])], axis=1))
            test_frame = h2o.H2OFrame(pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test, columns=['target'])], axis=1))
            
            # Define feature and target columns
            x = train_frame.columns
            x.remove('target')
            y = 'target'
            
            # Run AutoML
            print(f"Training H2O AutoML {'classifier' if is_classification else 'regressor'}...")
            aml = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                max_models=max_models,
                seed=42
            )
            aml.train(x=x, y=y, training_frame=train_frame)
            
            # Model leaderboard
            leaderboard = aml.leaderboard
            print("\nModel Leaderboard:")
            print(leaderboard.head())
            
            # Make predictions
            preds = aml.predict(test_frame)
            
            # Convert predictions to numpy array
            if is_classification:
                y_pred = preds['predict'].as_data_frame().values.ravel()
                
                # Evaluate
                from sklearn.metrics import accuracy_score, f1_score
                results = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
            else:
                y_pred = preds.as_data_frame().values.ravel()
                
                # Evaluate
                from sklearn.metrics import mean_squared_error, r2_score
                results = {
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
            
            return {
                'model': aml,
                'predictions': y_pred,
                'results': results,
                'leaderboard': leaderboard
            }
            
        except ImportError:
            print("H2O not installed. Install with: pip install h2o")
            return None
    
    @staticmethod
    def compare_automl_platforms(X_train, y_train, X_test, y_test, is_classification=True, time_budget=1800):
        """
        Compare different AutoML platforms
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            is_classification: Whether it's a classification task
            time_budget: Time budget in seconds
            
        Returns:
            results: Dictionary with results from each platform
        """
        results = {}
        
        # AutoML platforms to try
        platforms = [
            'auto_sklearn',
            'tpot',
            'h2o'
        ]
        
        for platform in platforms:
            print(f"\n===== Testing {platform} =====")
            
            try:
                if platform == 'auto_sklearn':
                    if is_classification:
                        result = AutoML.auto_sklearn_classification(X_train, y_train, X_test, y_test, time_left=time_budget)
                    else:
                        result = AutoML.auto_sklearn_regression(X_train, y_train, X_test, y_test, time_left=time_budget)
                
                elif platform == 'tpot':
                    result = AutoML.tpot_classifier(X_train, y_train, X_test, y_test, max_time_mins=time_budget//60)
                
                elif platform == 'h2o':
                    result = AutoML.h2o_automl(X_train, y_train, X_test, y_test, max_runtime_secs=time_budget, is_classification=is_classification)
                
                results[platform] = result
                
            except Exception as e:
                print(f"Error testing {platform}: {str(e)}")
                results[platform] = None
        
        return results


def main():
    """Main function to demonstrate Classical ML Milestone Project"""
    print("Classical ML Milestone Project - Comprehensive Machine Learning Pipeline")
    print("=" * 70)
    
    # Example: Load and prepare data (replace with your dataset)
    print("\nStep 1: Data Preparation")
    # You can use one of these datasets for testing:
    # - Iris dataset for classification
    # - Boston Housing for regression
    # - Wine dataset for clustering
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset: Iris with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Demonstrate mathematical foundations
    print("\nStep 2: Mathematical Foundations")
    math_foundations = MathFoundations()
    linear_algebra_results = math_foundations.linear_algebra_operations()
    print("Linear Algebra Operations:")
    print(f"- Vector addition: {linear_algebra_results['vector_addition']}")
    print(f"- Matrix multiplication:\n{linear_algebra_results['matrix_multiplication']}")
    print(f"- Eigenvalues: {linear_algebra_results['eigenvalues']}")
    
    # Feature engineering
    print("\nStep 3: Feature Engineering")
    X_train_scaled, X_test_scaled, scaler = FeatureEngineering.standardize_features(X_train, X_test)
    print("Features standardized successfully")
    
    # Train classification models
    print("\nStep 4: Model Training")
    supervised = SupervisedLearning()
    models = supervised.train_classification_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("Classification Results:")
    for name, results in models.items():
        print(f"- {name}: Accuracy = {results['test_accuracy']:.4f}, F1 = {results['test_f1']:.4f}")
    
    # Create ensemble model
    voting_clf = supervised.create_voting_classifier(models, X_train_scaled, y_train)
    voting_pred = voting_clf.predict(X_test_scaled)
    voting_accuracy = accuracy_score(y_test, voting_pred)
    print(f"- Voting Classifier: Accuracy = {voting_accuracy:.4f}")
    
    # Unsupervised learning
    print("\nStep 5: Unsupervised Learning")
    unsupervised = UnsupervisedLearning()
    pca_result = unsupervised.pca_dimension_reduction(X, n_components=2)
    print(f"PCA explained variance: {pca_result['explained_variance_ratio']}")
    
    kmeans_result = unsupervised.kmeans_clustering(X, n_clusters=3)
    print(f"K-means inertia: {kmeans_result['inertia']}")
    
    print("\nProject setup complete! You can now enhance and extend the code for your specific needs.")


if __name__ == "__main__":
    main()
