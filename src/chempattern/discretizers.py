"""Feature discretization transformers.

This module provides scikit-learn compatible transformers for converting
continuous features into discrete categories suitable for pattern mining.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class BaseDiscretizer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for scikit-learn compatible discretizers.
    
    All discretizers follow the scikit-learn transformer API with fit()
    and transform() methods. They convert continuous features into
    boolean one-hot encoded categories.
    
    Attributes:
        cutoffs_ (dict): Dictionary mapping feature names to cutoff values
        feature_names_in_ (list): List of input feature names
    """
    
    def __init__(self):
        self.cutoffs_ = {}
        self.feature_names_in_ = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the discretizer to the data.
        
        Args:
            X (pd.DataFrame): Features to discretize
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self: The fitted discretizer
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        """Transform features into discrete categories.
        
        Args:
            X (pd.DataFrame): Features to discretize
            
        Returns:
            pd.DataFrame: One-hot encoded boolean features
        """
        raise NotImplementedError

    def _df_to_transactions(self, df_one_hot: pd.DataFrame) -> list:
        """Convert a one-hot boolean DataFrame to a PAMI transaction list.
        
        This method converts a boolean DataFrame (where True indicates
        presence of a feature) into the transaction format required by
        PAMI mining algorithms.
        
        Args:
            df_one_hot (pd.DataFrame): Boolean DataFrame with one-hot encoding
            
        Returns:
            list: List of lists, where each inner list contains the names
                of features that are True for that row
                
        Example:
            >>> df = pd.DataFrame({'A': [True, False], 'B': [False, True]})
            >>> discretizer._df_to_transactions(df)
            [['A'], ['B']]
        """
        transactions = []
        for i in range(len(df_one_hot)):
            transactions.append(
                list(df_one_hot.columns[df_one_hot.iloc[i] == True])
            )
        return transactions


class MedianDiscretizer(BaseDiscretizer):
    """Discretize features into 'Above_Median' and 'Below_Median' categories.
    
    This discretizer splits each feature at its median value, creating
    two boolean features for each input feature.
    
    Example:
        >>> from chempattern.discretizers import MedianDiscretizer
        >>> import pandas as pd
        >>> X = pd.DataFrame({'MolWt': [100, 200, 300], 'LogP': [1, 2, 3]})
        >>> disc = MedianDiscretizer()
        >>> disc.fit(X)
        >>> X_discrete = disc.transform(X)
        >>> print(X_discrete.columns)
        Index(['MolWt_Above_Median', 'MolWt_Below_Median',
               'LogP_Above_Median', 'LogP_Below_Median'], dtype='object')
    """
    
    def fit(self, X, y=None):
        """Fit by computing median for each feature.
        
        Args:
            X (pd.DataFrame): Features to discretize
            y: Ignored
            
        Returns:
            self: The fitted discretizer
        """
        self.feature_names_in_ = list(X.columns)
        for col in self.feature_names_in_:
            self.cutoffs_[col] = X[col].median()
        return self

    def transform(self, X):
        """Transform features into above/below median categories.
        
        Args:
            X (pd.DataFrame): Features to discretize
            
        Returns:
            pd.DataFrame: Boolean DataFrame with _Above_Median and 
                _Below_Median columns for each feature
                
        Raises:
            RuntimeError: If discretizer has not been fitted
        """
        if not self.cutoffs_:
            self.logger.error("Discretizer has not been fitted.")
            raise RuntimeError("Discretizer has not been fitted.")
        
        df_tx_list = []
        for col in self.feature_names_in_:
            df_tx_col = pd.DataFrame(index=X.index)
            median_val = self.cutoffs_[col]
            df_tx_col[f'{col}_Above_Median'] = (X[col] > median_val)
            df_tx_col[f'{col}_Below_Median'] = (X[col] <= median_val)
            df_tx_list.append(df_tx_col)
            
        return pd.concat(df_tx_list, axis=1)


class QuantileDiscretizer(BaseDiscretizer):
    """Discretize features into quantile-based categories.
    
    This discretizer splits each feature into equal-frequency bins
    based on quantiles. By default, creates 'Low', 'Med', 'High' bins.
    
    Attributes:
        num_bins (int): Number of bins to create
        labels (list): Labels for each bin
        
    Example:
        >>> from chempattern.discretizers import QuantileDiscretizer
        >>> import pandas as pd
        >>> X = pd.DataFrame({'MolWt': range(100)})
        >>> disc = QuantileDiscretizer(num_bins=3)
        >>> disc.fit(X)
        >>> X_discrete = disc.transform(X)
        >>> print(X_discrete.columns)
        Index(['MolWt_Low', 'MolWt_Med', 'MolWt_High'], dtype='object')
    """
    
    def __init__(self, num_bins=3):
        """Initialize the quantile discretizer.
        
        Args:
            num_bins (int): Number of quantile bins to create. Default is 3
                for Low/Med/High categories.
        """
        super().__init__()
        self.num_bins = num_bins
        self.labels = (
            ['Low', 'Med', 'High'] if num_bins == 3 
            else [f'Q{i}' for i in range(num_bins)]
        )

    def fit(self, X, y=None):
        """Fit by computing quantile bin edges for each feature.
        
        Args:
            X (pd.DataFrame): Features to discretize
            y: Ignored
            
        Returns:
            self: The fitted discretizer
            
        Note:
            For features with little variation, falls back to min/max bins
        """
        self.feature_names_in_ = list(X.columns)
        for col in self.feature_names_in_:
            try:
                # Store the bin edges
                _, self.cutoffs_[col] = pd.qcut(
                    X[col], 
                    self.num_bins, 
                    retbins=True, 
                    duplicates='drop'
                )
            except Exception:
                # Fallback for constant columns: store single value
                self.cutoffs_[col] = [X[col].min(), X[col].max()]
        return self

    def transform(self, X):
        """Transform features into quantile-based categories.
        
        Args:
            X (pd.DataFrame): Features to discretize
            
        Returns:
            pd.DataFrame: Boolean one-hot encoded DataFrame with quantile
                categories for each feature
                
        Raises:
            RuntimeError: If discretizer has not been fitted
        """
        if not self.cutoffs_:
            self.logger.error("Discretizer has not been fitted.")
            raise RuntimeError("Discretizer has not been fitted.")
        
        df_tx_list = []
        for col in self.feature_names_in_:
            try:
                labels = [
                    f'{col}_{l}' 
                    for l in self.labels[:len(self.cutoffs_[col])-1]
                ]
                binned = pd.cut(
                    X[col], 
                    bins=self.cutoffs_[col], 
                    labels=labels, 
                    include_lowest=True
                )
                df_tx_list.append(pd.get_dummies(binned))
            except Exception as e:
                self.logger.warning(
                    f"Could not transform feature '{col}'. Skipping. Error: {e}"
                )
                
        return pd.concat(df_tx_list, axis=1)
