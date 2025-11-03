"""Data loading and feature generation module.

This module handles loading chemical data, merging feature files,
and generating RDKit molecular descriptors.
"""

import pandas as pd
import numpy as np
import logging
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.error')


class ChemDataManager:
    """Handles loading, merging, and RDKit feature generation.
    
    This class manages the entire data loading pipeline, including:
    - Loading main CSV files with SMILES and endpoints
    - Optional merging of pre-computed feature files
    - Automatic generation of RDKit molecular descriptors
    - Detection and validation of endpoint columns
    
    Attributes:
        id_col (str): Name of the ID column
        smiles_col (str): Name of the SMILES column
        feature_cols (list): List of feature column names
        endpoints_to_analyze (list): List of endpoint column names
        data_df (pd.DataFrame): Main dataframe with all data
        
    Example:
        >>> manager = ChemDataManager(
        ...     main_file='compounds.csv',
        ...     id_col='compound_id',
        ...     smiles_col='smiles',
        ...     target_endpoints=['IC50']
        ... )
        >>> data = manager.get_data_for_endpoint('IC50')
    """
    
    def __init__(self, main_file: str, id_col: str, smiles_col: str, 
                 feature_file: str = None, target_endpoints: list = None):
        """Initialize the ChemDataManager.
        
        Args:
            main_file (str): Path to main CSV file with SMILES and endpoints
            id_col (str): Name of the ID column
            smiles_col (str): Name of the SMILES column
            feature_file (str, optional): Path to pre-computed features CSV
            target_endpoints (list, optional): List of specific endpoints to analyze.
                If None, all numeric columns (except ID and features) will be used.
                
        Raises:
            FileNotFoundError: If main_file doesn't exist
            ValueError: If required columns are missing
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing ChemDataManager for {main_file}...")
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.feature_cols = []
        
        try:
            self.main_df = pd.read_csv(main_file)
        except FileNotFoundError:
            self.logger.error(f"Main file not found at {main_file}")
            raise

        if not all(col in self.main_df.columns for col in [id_col, smiles_col]):
            raise ValueError(f"Main file must contain '{id_col}' and '{smiles_col}' columns.")

        if feature_file:
            self._load_features(feature_file)
        else:
            self.logger.info("No feature file provided. Generating RDKit features...")
            self._generate_rdkit_features()

        self._determine_endpoints(target_endpoints)
        self.logger.info(
            f"Ready. {len(self.feature_cols)} features loaded. "
            f"{len(self.endpoints_to_analyze)} endpoints to analyze."
        )

    def _load_features(self, feature_file: str):
        """Load and merge a pre-computed feature file.
        
        Args:
            feature_file (str): Path to CSV file with features
            
        Note:
            If loading fails, will fall back to generating RDKit features.
        """
        self.logger.info(f"Loading features from {feature_file}...")
        try:
            feature_df = pd.read_csv(feature_file)
            if self.id_col not in feature_df.columns:
                raise ValueError(f"Feature file must contain '{self.id_col}' for merging.")
            
            self.data_df = pd.merge(self.main_df, feature_df, on=self.id_col, how='left')
            self.feature_cols = [col for col in feature_df.columns if col != self.id_col]
            self.logger.info(f"Successfully merged {len(self.feature_cols)} features.")
        except FileNotFoundError:
            self.logger.warning(
                f"Feature file not found at {feature_file}. Will generate features."
            )
            self._generate_rdkit_features()

    def _generate_rdkit_features(self):
        """Generate a default set of RDKit molecular descriptors.
        
        Computes the following descriptors for each molecule:
        - MolWt: Molecular weight
        - LogP: Partition coefficient
        - TPSA: Topological polar surface area
        - NumHDonors: Number of hydrogen bond donors
        - NumHAcceptors: Number of hydrogen bond acceptors
        - NumRotatableBonds: Number of rotatable bonds
        - NumAromaticRings: Number of aromatic rings
        - FractionCSP3: Fraction of sp3 carbons
        
        Invalid SMILES will have NaN values for all descriptors.
        """
        descriptors_to_calc = [
            ('MolWt', Descriptors.MolWt),
            ('LogP', Descriptors.MolLogP),
            ('TPSA', Descriptors.TPSA),
            ('NumHDonors', Descriptors.NumHDonors),
            ('NumHAcceptors', Descriptors.NumHAcceptors),
            ('NumRotatableBonds', Descriptors.NumRotatableBonds),
            ('NumAromaticRings', Descriptors.NumAromaticRings),
            ('FractionCSP3', Descriptors.FractionCSP3)
        ]
        
        self.feature_cols = [name for name, func in descriptors_to_calc]
        features_list = []

        # Create data_df if it doesn't exist (from main_df)
        if not hasattr(self, 'data_df'):
            self.data_df = self.main_df.copy()

        for smiles in self.data_df[self.smiles_col]:
            mol = Chem.MolFromSmiles(str(smiles))
            features = {}
            if mol:
                for name, func in descriptors_to_calc:
                    try:
                        features[name] = func(mol)
                    except Exception:
                        features[name] = np.nan
            else:
                for name, func in descriptors_to_calc:
                    features[name] = np.nan
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list, index=self.data_df.index)
        self.data_df = pd.concat([self.data_df, features_df], axis=1)
        self.logger.info(f"Generated {len(self.feature_cols)} RDKit features.")

    def _determine_endpoints(self, target_endpoints):
        """Identify the endpoint columns to be analyzed.
        
        Args:
            target_endpoints (list or None): Specific endpoints to use, or None
                to auto-detect all numeric columns
                
        Raises:
            ValueError: If specified endpoints don't exist in data
        """
        if target_endpoints:
            missing = [e for e in target_endpoints if e not in self.data_df.columns]
            if missing:
                raise ValueError(f"Specified endpoints not in data: {missing}")
            self.endpoints_to_analyze = target_endpoints
        else:
            potential_endpoints = self.data_df.select_dtypes(include=np.number).columns
            self.endpoints_to_analyze = [
                e for e in potential_endpoints 
                if e != self.id_col and e not in self.feature_cols
            ]
        
        # Ensure all endpoints are numeric
        for col in self.endpoints_to_analyze:
            self.data_df[col] = pd.to_numeric(self.data_df[col], errors='coerce')

    def get_data_for_endpoint(self, endpoint: str) -> pd.DataFrame:
        """Return a clean DataFrame for a single endpoint analysis.
        
        This method:
        1. Selects relevant columns (ID, SMILES, endpoint, features)
        2. Drops rows with NaN values in endpoint or features
        3. Returns only numeric feature columns and the endpoint
        
        Args:
            endpoint (str): Name of the endpoint column
            
        Returns:
            pd.DataFrame: Cleaned dataframe with features and endpoint
            
        Example:
            >>> data = manager.get_data_for_endpoint('IC50')
            >>> print(data.columns)
            Index(['MolWt', 'LogP', 'TPSA', ..., 'IC50'], dtype='object')
        """
        cols_to_use = [self.id_col, self.smiles_col, endpoint] + self.feature_cols
        # Ensure we don't have duplicate columns
        cols_to_use = list(dict.fromkeys(cols_to_use))
        
        clean_df = self.data_df[cols_to_use].dropna(
            subset=[endpoint] + self.feature_cols
        ).copy()
        
        features_df = clean_df[self.feature_cols].select_dtypes(include=np.number)
        endpoint_sr = clean_df[endpoint]
        
        return pd.concat([features_df, endpoint_sr], axis=1)

    def get_endpoints_to_analyze(self) -> list:
        """Return the list of endpoints that will be analyzed.
        
        Returns:
            list: List of endpoint column names
        """
        return self.endpoints_to_analyze
