import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import re
import string
from collections import Counter
import io
import base64

class DataAnalysisTool:
    """
    A standalone tool for data analysis that doesn't require external APIs.
    """
    def __init__(self):
        self.data = None
        self.original_data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.cleaned = False
        self.analysis_results = {}
    
    def load_data(self, file_content, file_type='csv'):
        """
        Load data from file content
        
        Parameters:
        -----------
        file_content : bytes or file-like object
            The content of the uploaded file
        file_type : str
            The file type ('csv' or 'excel')
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            if file_type == 'csv':
                # Try with different delimiters
                for delimiter in [',', ';', '\t', '|']:
                    try:
                        self.data = pd.read_csv(io.BytesIO(file_content), delimiter=delimiter)
                        break
                    except:
                        continue
            elif file_type in ['xlsx', 'xls', 'excel']:
                self.data = pd.read_excel(io.BytesIO(file_content))
            else:
                return False, "Unsupported file type"
            
            if self.data is None:
                return False, "Failed to load data"
                
            # Make a copy of original data
            self.original_data = self.data.copy()
            
            # Identify numeric and categorical columns
            self.numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Reset state
            self.cleaned = False
            self.analysis_results = {}
            
            return True, f"Data loaded successfully. Shape: {self.data.shape}"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def clean_data(self, remove_duplicates=True, handle_missing='fill'):
        """
        Clean the dataset by:
        - Removing duplicates
        - Handling missing values
        - Standardizing column names
        
        Parameters:
        -----------
        remove_duplicates : bool
            Whether to remove duplicate rows
        handle_missing : str
            How to handle missing values ('fill', 'drop_rows', 'drop_cols')
            
        Returns:
        --------
        dict
            Information about the cleaning operations performed
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        # Initialize cleaning info
        cleaning_info = {
            'original_shape': self.data.shape,
            'duplicates_removed': 0,
            'missing_values_handled': {},
            'columns_renamed': {},
        }
        
        # 1. Standardize column names
        original_columns = self.data.columns.tolist()
        
        # Function to clean column names
        def clean_column_name(col):
            col = str(col).strip().lower()
            col = re.sub(r'[^a-z0-9]', '_', col)
            col = re.sub(r'_+', '_', col)
            col = col.strip('_')
            return col if col else 'column'
        
        # Apply cleaning to column names
        self.data.columns = [clean_column_name(col) for col in self.data.columns]
        cleaning_info['columns_renamed'] = dict(zip(original_columns, self.data.columns.tolist()))
        
        # 2. Remove duplicates if requested
        if remove_duplicates:
            duplicate_count = self.data.duplicated().sum()
            if duplicate_count > 0:
                self.data = self.data.drop_duplicates().reset_index(drop=True)
                cleaning_info['duplicates_removed'] = int(duplicate_count)
        
        # 3. Handle missing values
        missing_values = self.data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        if not columns_with_missing.empty:
            if handle_missing == 'fill':
                for col in columns_with_missing.index:
                    # For numeric columns, fill with mean or median
                    if col in self.numeric_columns:
                        # Check if skewed
                        if abs(self.data[col].skew()) > 1:
                            self.data[col] = self.data[col].fillna(self.data[col].median())
                            method = 'median'
                        else:
                            self.data[col] = self.data[col].fillna(self.data[col].mean())
                            method = 'mean'
                    else:
                        # For non-numeric, fill with mode
                        mode_value = self.data[col].mode()[0] if not self.data[col].mode().empty else 'Unknown'
                        self.data[col] = self.data[col].fillna(mode_value)
                        method = 'mode'
                    
                    cleaning_info['missing_values_handled'][col] = {
                        'original_missing': int(columns_with_missing[col]),
                        'method': method
                    }
            
            elif handle_missing == 'drop_rows':
                rows_before = len(self.data)
                self.data = self.data.dropna()
                rows_removed = rows_before - len(self.data)
                cleaning_info['missing_values_handled']['rows_dropped'] = rows_removed
            
            elif handle_missing == 'drop_cols':
                # Drop columns with more than 50% missing values
                cols_to_drop = [col for col in columns_with_missing.index 
                               if columns_with_missing[col] / len(self.data) >= 0.5]
                if cols_to_drop:
                    self.data = self.data.drop(columns=cols_to_drop)
                    cleaning_info['missing_values_handled']['columns_dropped'] = cols_to_drop
        
        # 4. Update column lists after cleaning
        self.numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 5. Update cleaning info with final shape
        cleaning_info['final_shape'] = self.data.shape
        self.cleaned = True
        
        return cleaning_info
    
    def get_data_overview(self):
        """
        Get overview statistics of the data
        
        Returns:
        --------
        dict
            Dictionary with data overview information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        overview = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.apply(lambda x: str(x)).to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
        }
        
        # Add descriptive statistics for numeric columns
        if self.numeric_columns:
            overview['numeric_stats'] = self.data[self.numeric_columns].describe().to_dict()
        
        # Add value counts for categorical columns (up to 10 most frequent)
        if self.categorical_columns:
            cat_stats = {}
            for col in self.categorical_columns:
                value_counts = self.data[col].value_counts().head(10).to_dict()
                unique_count = self.data[col].nunique()
                cat_stats[col] = {
                    'top_values': value_counts,
                    'unique_count': unique_count
                }
            overview['categorical_stats'] = cat_stats
        
        return overview
    
    def perform_eda(self):
        """
        Perform exploratory data analysis
        
        Returns:
        --------
        dict
            Dictionary with EDA results
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        eda_results = {
            'summary_statistics': {},
            'correlation_matrix': None,
            'skewness': {},
            'outliers': {},
        }
        
        # Descriptive statistics
        if self.numeric_columns:
            eda_results['summary_statistics'] = self.data[self.numeric_columns].describe().to_dict()
            
            # Correlation matrix
            eda_results['correlation_matrix'] = self.data[self.numeric_columns].corr().to_dict()
            
            # Skewness
            eda_results['skewness'] = self.data[self.numeric_columns].skew().to_dict()
            
            # Detect outliers using IQR method
            outliers = {}
            for col in self.numeric_columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float(outlier_count / len(self.data) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
            
            eda_results['outliers'] = outliers
        
        # Value distributions for categorical columns
        if self.categorical_columns:
            cat_distributions = {}
            for col in self.categorical_columns:
                value_counts = self.data[col].value_counts().head(10).to_dict()
                cat_distributions[col] = value_counts
            
            eda_results['categorical_distributions'] = cat_distributions
        
        # Store in analysis results
        self.analysis_results['eda'] = eda_results
        
        return eda_results
    
    def generate_visualizations(self):
        """
        Generate base64-encoded visualization plots
        
        Returns:
        --------
        dict
            Dictionary with visualization data
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        visualization_data = {
            'histograms': {},
            'boxplots': {},
            'correlation_heatmap': None,
            'scatter_plots': {},
            'bar_charts': {}
        }
        
        # Histograms for numeric columns
        for col in self.numeric_columns[:5]:  # Limit to 5 columns
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            visualization_data['histograms'][col] = img_str
        
        # Boxplots for numeric columns
        for col in self.numeric_columns[:5]:  # Limit to 5 columns
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=self.data[col])
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            visualization_data['boxplots'][col] = img_str
        
        # Correlation heatmap
        if len(self.numeric_columns) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.data[self.numeric_columns].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=.5, fmt='.2f')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            visualization_data['correlation_heatmap'] = img_str
        
        # Scatter plots for pairs of numeric columns
        if len(self.numeric_columns) >= 2:
            # Get pairs of columns (limit to first 3 columns to avoid too many plots)
            cols_for_scatter = self.numeric_columns[:3]
            for i in range(len(cols_for_scatter)):
                for j in range(i+1, len(cols_for_scatter)):
                    col1, col2 = cols_for_scatter[i], cols_for_scatter[j]
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=self.data[col1], y=self.data[col2])
                    plt.title(f'{col2} vs {col1}')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.tight_layout()
                    
                    # Save to buffer and convert to base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.read()).decode()
                    plt.close()
                    
                    visualization_data['scatter_plots'][f'{col1}_vs_{col2}'] = img_str
        
        # Bar charts for categorical columns
        for col in self.categorical_columns[:5]:  # Limit to 5 columns
            # Skip if too many unique values
            if self.data[col].nunique() > 20:
                continue
                
            plt.figure(figsize=(12, 6))
            value_counts = self.data[col].value_counts().head(10)  # Top 10 categories
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Count of {col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            visualization_data['bar_charts'][col] = img_str
        
        # Store in analysis results
        self.analysis_results['visualizations'] = visualization_data
        
        return visualization_data
    
    def build_model(self, target_column=None, model_type='auto', test_size=0.2):
        """
        Build a predictive model based on the data
        
        Parameters:
        -----------
        target_column : str or None
            Target column for supervised learning (None for clustering)
        model_type : str
            Type of model to build ('auto', 'linear_regression', 'random_forest', 'kmeans')
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Model results
        """
        if self.data is None:
            return {"error": "No data loaded"}
            
        model_results = {}
        
        # For clustering (unsupervised learning)
        if target_column is None or model_type == 'kmeans':
            # Perform K-means clustering on numeric data
            if len(self.numeric_columns) < 2:
                return {"error": "Need at least 2 numeric columns for clustering"}
            
            # Select only numeric columns and drop rows with NaN
            X = self.data[self.numeric_columns].dropna()
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal k using the elbow method
            inertia_values = []
            k_values = range(1, min(11, len(X)))
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia_values.append(kmeans.inertia_)
            
            # Find elbow point (approximate)
            inertia_diff = np.diff(inertia_values)
            elbow_idx = np.argmax(np.diff(inertia_diff)) + 1 if len(inertia_diff) > 1 else 1
            optimal_k = k_values[min(elbow_idx, len(k_values)-1)]
            
            # Fit model with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to original data
            X_with_clusters = X.copy()
            X_with_clusters['cluster'] = cluster_labels
            
            # Calculate cluster statistics
            cluster_stats = X_with_clusters.groupby('cluster').mean().to_dict()
            
            # Create cluster visualization using PCA if more than 2 dimensions
            if len(self.numeric_columns) > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Create scatter plot
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
                plt.title('Cluster Assignments (PCA)')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                
                # Save to buffer and convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                cluster_plot = base64.b64encode(buffer.read()).decode()
                plt.close()
            else:
                # Use the first two columns directly
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X[self.numeric_columns[0]], X[self.numeric_columns[1]], 
                                     c=cluster_labels, cmap='viridis')
                plt.title('Cluster Assignments')
                plt.xlabel(self.numeric_columns[0])
                plt.ylabel(self.numeric_columns[1])
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                
                # Save to buffer and convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                cluster_plot = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            # Create elbow curve
            plt.figure(figsize=(10, 6))
            plt.plot(list(k_values), inertia_values, 'bo-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.axvline(x=optimal_k, color='r', linestyle='--', 
                      label=f'Optimal k = {optimal_k}')
            plt.legend()
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            elbow_plot = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # Store results
            model_results = {
                'model_type': 'kmeans',
                'optimal_k': optimal_k,
                'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict(),
                'cluster_stats': cluster_stats,
                'visualization': {
                    'cluster_plot': cluster_plot,
                    'elbow_plot': elbow_plot
                }
            }
            
        # For supervised learning
        elif target_column is not None:
            if target_column not in self.data.columns:
                return {"error": f"Target column '{target_column}' not found in data"}
            
            # Prepare features and target
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Handle categorical features (one-hot encoding)
            X = pd.get_dummies(X, drop_first=True)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Determine if regression or classification
            is_categorical = y.dtype == 'object' or y.dtype == 'category' or y.nunique() < 10
            
            # Choose appropriate model
            if is_categorical:  # Classification
                if model_type == 'auto' or model_type == 'random_forest':
                    # Random Forest Classifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    # Logistic Regression
                    model = LogisticRegression(random_state=42, max_iter=1000)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                
                # Save to buffer and convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                cm_plot = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                # Feature importance plot (if Random Forest)
                if isinstance(model, RandomForestClassifier):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    plt.figure(figsize=(10, 6))
                    plt.title('Feature Importances')
                    plt.bar(range(min(10, len(importances))), 
                           importances[indices[:10]],
                           align='center')
                    plt.xticks(range(min(10, len(importances))), 
                              [X.columns[i] for i in indices[:10]], 
                              rotation=90)
                    plt.tight_layout()
                    
                    # Save to buffer and convert to base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    importance_plot = base64.b64encode(buffer.read()).decode()
                    plt.close()
                else:
                    importance_plot = None
                
                # Store results
                model_results = {
                    'model_type': 'random_forest' if isinstance(model, RandomForestClassifier) else 'logistic_regression',
                    'task_type': 'classification',
                    'target_column': target_column,
                    'metrics': {
                        'train_accuracy': float(train_accuracy),
                        'test_accuracy': float(test_accuracy)
                    },
                    'visualization': {
                        'confusion_matrix': cm_plot,
                        'feature_importance': importance_plot
                    }
                }
                
            else:  # Regression
                if model_type == 'auto' or model_type == 'random_forest':
                    # Random Forest Regressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    # Linear Regression
                    model = LinearRegression()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Create scatter plot of actual vs predicted
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_test_pred, alpha=0.5)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted Values')
                plt.tight_layout()
                
                # Save to buffer and convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                pred_plot = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                # Feature importance plot (if Random Forest)
                if isinstance(model, RandomForestRegressor):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    plt.figure(figsize=(10, 6))
                    plt.title('Feature Importances')
                    plt.bar(range(min(10, len(importances))), 
                           importances[indices[:10]],
                           align='center')
                    plt.xticks(range(min(10, len(importances))), 
                              [X.columns[i] for i in indices[:10]], 
                              rotation=90)
                    plt.tight_layout()
                    
                    # Save to buffer and convert to base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    importance_plot = base64.b64encode(buffer.read()).decode()
                    plt.close()
                else:
                    # Coefficient plot for linear regression
                    coefs = model.coef_
                    indices = np.argsort(np.abs(coefs))[::-1]
                    
                    plt.figure(figsize=(10, 6))
                    plt.title('Coefficient Magnitudes')
                    plt.bar(range(min(10, len(coefs))), 
                           np.abs(coefs[indices[:10]]),
                           align='center')
                    plt.xticks(range(min(10, len(coefs))), 
                              [X.columns[i] for i in indices[:10]], 
                              rotation=90)
                    plt.tight_layout()
                    
                    # Save to buffer and convert to base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    importance_plot = base64.b64encode(buffer.read()).decode()
                    plt.close()
                
                # Store results
                model_results = {
                    'model_type': 'random_forest' if isinstance(model, RandomForestRegressor) else 'linear_regression',
                    'task_type': 'regression',
                    'target_column': target_column,
                    'metrics': {
                        'train_rmse': float(train_rmse),
                        'test_rmse': float(test_rmse),
                        'train_r2': float(train_r2),
                        'test_r2': float(test_r2)
                    },
                    'visualization': {
                        'prediction_plot': pred_plot,
                        'feature_importance': importance_plot
                    }
                }
        
        # Store in analysis results
        self.analysis_results['model'] = model_results
        
        return model_results
    
    def get_data_summary(self):
        """
        Generate a comprehensive summary of the data analysis
        
        Returns:
        --------
        dict
            Summary of findings
        """
        if self.data is None:
            return {"error": "No data loaded"}
            
        if not self.analysis_results:
            return {"error": "No analysis has been performed yet"}
            
        summary = {
            "dataset": {
                "shape": self.data.shape,
                "columns": len(self.data.columns),
                "numeric_columns": len(self.numeric_columns),
                "categorical_columns": len(self.categorical_columns),
                "missing_values": self.data.isna().sum().sum()
            },
            "key_insights": []
        }
        
        # Add insights about data quality
        if self.cleaned:
            summary["key_insights"].append({
                "type": "data_quality",
                "title": "Data Quality",
                "details": f"The dataset had {summary['dataset']['missing_values']} missing values across all columns."
            })
        
        # Add insights from EDA
        if 'eda' in self.analysis_results:
            eda = self.analysis_results['eda']
            
            # Correlation insights
            if 'correlation_matrix' in eda and eda['correlation_matrix']:
                corr_matrix = pd.DataFrame(eda['correlation_matrix'])
                # Find strongest correlations
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.7:  # Strong correlation threshold
                            high_corr.append((col1, col2, corr_val))
                
                if high_corr:
                    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
                    corr_details = [f"{col1} and {col2} (r={corr:.2f})" for col1, col2, corr in high_corr[:3]]
                    summary["key_insights"].append({
                        "type": "correlation",
                        "title": "Strong Correlations",
                        "details": "Found strong correlations between: " + ", ".join(corr_details)
                    })
            
            # Outlier insights
            if 'outliers' in eda and eda['outliers']:
                outlier_cols = list(eda['outliers'].keys())
                if outlier_cols:
                    # Get top 3 columns with highest percentage of outliers
                    top_outliers = sorted(outlier_cols, 
                                        key=lambda x: eda['outliers'][x]['percentage'], 
                                        reverse=True)[:3]
                    outlier_details = [f"{col} ({eda['outliers'][col]['percentage']:.1f}%)" for col in top_outliers]
                    summary["key_insights"].append({
                        "type": "outliers",
                        "title": "Outliers Detected",
                        "details": "Significant outliers found in: " + ", ".join(outlier_details)
                    })
            
            # Skewness insights
            if 'skewness' in eda and eda['skewness']:
                skewed_cols = {col: skew for col, skew in eda['skewness'].items() if abs(skew) > 1}
                if skewed_cols:
                    # Get top 3 most skewed columns
                    top_skewed = sorted(skewed_cols.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    skew_details = [f"{col} (skew={skew:.2f})" for col, skew in top_skewed]
                    summary["key_insights"].append({
                        "type": "skewness",
                        "title": "Skewed Distributions",
                        "details": "Highly skewed distributions in: " + ", ".join(skew_details)
                    })
        
        # Add insights from modeling
        if 'model' in self.analysis_results:
            model = self.analysis_results['model']
            
            if 'task_type' in model:
                if model['task_type'] == 'regression':
                    r2 = model['metrics'].get('test_r2', 0)
                    rmse = model['metrics'].get('test_rmse', 0)
                    summary["key_insights"].append({
                        "type": "model_performance",
                        "title": f"{model['model_type'].replace('_', ' ').title()} Model Performance",
                        "details": f"The model explains {r2*100:.1f}% of the variance in {model['target_column']} with RMSE of {rmse:.4f}."
                    })
                elif model['task_type'] == 'classification':
                    accuracy = model['metrics'].get('test_accuracy', 0)
                    summary["key_insights"].append({
                        "type": "model_performance",
                        "title": f"{model['model_type'].replace('_', ' ').title()} Model Performance",
                        "details": f"The model predicts {model['target_column']} with {accuracy*100:.1f}% accuracy."
                    })
            elif model['model_type'] == 'kmeans':
                summary["key_insights"].append({
                    "type": "clustering",
                    "title": "Clustering Results",
                    "details": f"Data was segmented into {model['optimal_k']} distinct clusters."
                })
        
        return summary
        
    def export_processed_data(self, format='csv'):
        """
        Export the processed dataset to CSV or Excel format
        
        Parameters:
        -----------
        format : str
            Export format ('csv' or 'excel')
            
        Returns:
        --------
        bytes
            The exported data as bytes
        """
        if self.data is None:
            return None
            
        buffer = io.BytesIO()
        
        if format == 'csv':
            self.data.to_csv(buffer, index=False)
        elif format in ['excel', 'xlsx']:
            self.data.to_excel(buffer, index=False)
        else:
            return None
            
        buffer.seek(0)
        return buffer.getvalue()