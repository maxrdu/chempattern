"""Analysis modules for pattern mining.

This module provides composable analyzers for different types of
pattern mining including exploratory analysis, contrast mining,
utility mining, and frequency mining.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from PAMI.frequentPattern.basic import FPGrowth as fpgrowth
from PAMI.AssociationRules.basic import confidence as post_process
from PAMI.highUtilityPattern.basic import EFIM as fhm

# TODO: erminer (contrast/emerging pattern mining) is not available in PAMI
# PAMI does not have a contrast pattern or emerging pattern mining module
# The ContrastAnalyzer will need to be refactored to use a different approach
# from PAMI.cep import erminer  # This module does not exist in PAMI

from .discretizers import BaseDiscretizer


def _write_transactions_to_file(transactions, filepath, sep='\t'):
    """Write transactions to a file in PAMI format.

    Args:
        transactions (list): List of transactions, where each transaction is a list of items
        filepath (str): Path to output file
        sep (str): Separator to use (default: tab)
    """
    with open(filepath, 'w') as f:
        for transaction in transactions:
            if transaction:  # Skip empty transactions
                f.write(sep.join(str(item) for item in transaction) + '\n')


def _write_utility_transactions_to_file(transactions, utilities, filepath):
    """Write transactions with utilities to a file in PAMI EFIM format.

    EFIM format: item1 item2 item3:total_utility:utility1 utility2 utility3

    Args:
        transactions (list): List of transactions, where each transaction is a list of items
        utilities (list): List of utilities (one per transaction)
        filepath (str): Path to output file
    """
    with open(filepath, 'w') as f:
        for transaction, total_utility in zip(transactions, utilities):
            if transaction:  # Skip empty transactions
                # For EFIM, we assign equal utility to all items in the transaction
                num_items = len(transaction)
                item_utility = int(total_utility / num_items) if num_items > 0 else 0
                items_str = ' '.join(str(item) for item in transaction)
                utilities_str = ' '.join(str(item_utility) for _ in transaction)
                f.write(f"{items_str}:{int(total_utility)}:{utilities_str}\n")


class BaseAnalyzer(ABC):
    """Abstract base class for all analysis types.
    
    Analyzers follow a consistent interface:
    1. Initialize with a descriptive name
    2. Implement run_and_plot() to perform analysis
    3. Return results and plot paths
    
    Attributes:
        name (str): Human-readable name for the analyzer
        results (dict): Storage for analysis results
    """
    
    def __init__(self, name: str):
        """Initialize the analyzer.
        
        Args:
            name (str): Descriptive name for this analyzer
        """
        self.name = name
        self.results = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run_and_plot(self, data: pd.DataFrame, tx_median: pd.DataFrame, 
                     tx_quantile: pd.DataFrame, endpoint: str, 
                     threshold: float, report_dir: str, **kwargs) -> tuple:
        """Run the analysis and generate plots.
        
        Args:
            data (pd.DataFrame): Cleaned data (features + endpoint)
            tx_median (pd.DataFrame): Median-discretized transactional data
            tx_quantile (pd.DataFrame): Quantile-discretized transactional data
            endpoint (str): Name of the endpoint column
            threshold (float): Binarization threshold for this endpoint
            report_dir (str): Directory to save plots
            **kwargs: Additional params (min_sup, min_conf, etc.)

        Returns:
            tuple: (results_data_dict, list_of_plot_paths)
                results_data_dict: Dict containing analysis results
                list_of_plot_paths: List of paths to generated plots
        """
        raise NotImplementedError


class EDAAnalyzer(BaseAnalyzer):
    """Perform exploratory data analysis and generate summary plots.
    
    This analyzer generates:
    1. Distribution plot for the endpoint
    2. Correlation heatmap showing feature-endpoint relationships
    
    No tabular results are produced.
    """
    
    def __init__(self):
        super().__init__("Exploratory Data Analysis")

    def run_and_plot(self, data: pd.DataFrame, tx_median: pd.DataFrame, 
                     tx_quantile: pd.DataFrame, endpoint: str, 
                     threshold: float, report_dir: str, **kwargs) -> tuple:
        """Generate EDA plots.
        
        Returns:
            tuple: (empty dict, list of 2 plot paths)
        """
        self.logger.info(f"Running {self.name}...")
        plot_paths = []
        
        # Plot 1: Endpoint Distribution
        plot_path_dist = os.path.join(report_dir, f"{endpoint}_dist_plot.png")
        plt.figure(figsize=(10, 6))
        sns.histplot(data[endpoint], kde=True)
        plt.axvline(
            threshold, 
            color='red', 
            linestyle='--', 
            label=f'Threshold ({threshold:.2f})'
        )
        plt.title(f'Distribution of {endpoint}', fontsize=16)
        plt.xlabel(endpoint)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path_dist)
        plt.close()
        plot_paths.append(plot_path_dist)
        
        # Plot 2: Correlation Heatmap
        plot_path_corr = os.path.join(report_dir, f"{endpoint}_corr_plot.png")
        plt.figure(figsize=(12, 8))
        
        # Scale features for a cleaner correlation
        scaled_features = StandardScaler().fit_transform(data.drop(columns=[endpoint]))
        scaled_features_df = pd.DataFrame(
            scaled_features, 
            columns=data.columns.drop(endpoint)
        )
        corr_data = pd.concat([scaled_features_df, data[endpoint]], axis=1)
        
        # Get correlation of all features with *just* the endpoint
        corr_map = (corr_data.corr()[[endpoint]]
                    .drop(endpoint)
                    .sort_values(by=endpoint, ascending=False))
        
        sns.heatmap(corr_map, annot=True, fmt=".2f", cmap="coolwarm_r")
        plt.title(f'Feature Correlation with {endpoint}', fontsize=16)
        plt.tight_layout()
        plt.savefig(plot_path_corr)
        plt.close()
        plot_paths.append(plot_path_corr)
        
        return {}, plot_paths  # EDA doesn't produce data tables


class ContrastAnalyzer(BaseAnalyzer):
    """Run Contrast Set / Emerging Pattern Mining (ERMiner).
    
    This analyzer identifies patterns that are significantly more frequent
    in high-activity molecules compared to low-activity molecules.
    
    Uses the ERMiner algorithm to find emerging patterns with high growth rates.
    """
    
    def __init__(self):
        super().__init__("Contrast Analysis")

    def run_and_plot(self, data: pd.DataFrame, tx_median: pd.DataFrame, 
                     tx_quantile: pd.DataFrame, endpoint: str, 
                     threshold: float, report_dir: str, **kwargs) -> tuple:
        """Run contrast pattern mining.
        
        Returns:
            tuple: (dict with 'data' DataFrame and 'threshold', list of plot paths)
        """
        self.logger.info(f"Running {self.name}...")
        
        data = data.copy()
        data['is_Positive'] = (data[endpoint] > threshold)
        
        tx_positive = tx_median[data['is_Positive'] == True]
        tx_negative = tx_median[data['is_Positive'] == False]
        
        # Convert to PAMI transactions
        tx_pos_list = BaseDiscretizer()._df_to_transactions(tx_positive)
        tx_neg_list = BaseDiscretizer()._df_to_transactions(tx_negative)
        
        if not tx_pos_list or not tx_neg_list:
            self.logger.warning("CEP: Skipping, one class has no transactions.")
            return {'data': pd.DataFrame(), 'threshold': threshold}, []

        # NOTE: ERMiner (contrast/emerging pattern mining) is not available in PAMI
        # PAMI does not currently have a contrast pattern or emerging pattern mining module
        # This analyzer is disabled until an alternative implementation is provided
        self.logger.warning(
            "CEP: Contrast pattern mining is not supported. "
            "PAMI does not have an ERMiner or emerging pattern module. "
            "This analysis is skipped."
        )
        return {'data': pd.DataFrame(), 'threshold': threshold}, []

        # # Original code (disabled - ERMiner doesn't exist in PAMI):
        # try:
        #     cep = erminer.ERMiner(
        #         tx_pos_list,
        #         tx_neg_list,
        #         min_sup=0.05,
        #         max_sup=0.5
        #     )
        #     cep.discover()
        #     emerging_patterns = cep.get_emerging_patterns()
        # except Exception as e:
        #     self.logger.error(f"CEP: ERMiner failed. Error: {e}")
        #     return {'data': pd.DataFrame(), 'threshold': threshold}, []
        #
        # cep_results = []
        # for pattern_str, supports in emerging_patterns.items():
        #     sup_pos = supports[0]
        #     sup_neg = supports[1]
        #     growth_rate = sup_pos / sup_neg if sup_neg > 0 else float('inf')
        #     cep_results.append({
        #         'itemset': pattern_str,
        #         'sup_Positive': sup_pos,
        #         'sup_Negative': sup_neg,
        #         'growth_rate': growth_rate
        #     })

        # # Rest of original code (disabled):
        # if not cep_results:
        #     self.logger.info("CEP: No emerging patterns found.")
        #     return {'data': pd.DataFrame(), 'threshold': threshold}, []
        #
        # cep_df = pd.DataFrame(cep_results)
        # cep_df = cep_df[cep_df['growth_rate'] >= 2].sort_values(
        #     'growth_rate',
        #     ascending=False
        # )
        #
        # plot_path = os.path.join(report_dir, f"{endpoint}_cep_plot.png")
        # cep_df_top10 = cep_df.head(10).sort_values('growth_rate', ascending=True)
        #
        # plt.figure(figsize=(10, 7))
        # sns.barplot(
        #     data=cep_df_top10,
        #     x='growth_rate',
        #     y='itemset',
        #     palette='rocket'
        # )
        # plt.title(
        #     f"Top 10 'Emerging' Patterns for {endpoint} "
        #     f"(Activity > {threshold:.2f})",
        #     fontsize=16
        # )
        # plt.xlabel(
        #     "Growth Rate (How much more frequent in 'Positive' class)",
        #     fontsize=12
        # )
        # plt.ylabel('Feature Pattern', fontsize=12)
        # plt.tight_layout()
        # plt.savefig(plot_path)
        # plt.close()
        #
        # return {'data': cep_df, 'threshold': threshold}, [plot_path]


class UtilityAnalyzer(BaseAnalyzer):
    """Run High-Utility Itemset Mining (FHM).
    
    This analyzer identifies patterns that contribute the most to the
    total endpoint value across the dataset. Patterns are weighted by
    their associated endpoint values.
    
    Uses the FHM (Fast High-Utility Miner) algorithm.
    """

    def __init__(self):
        super().__init__("High-Utility Analysis")

    def run_and_plot(self, data: pd.DataFrame, tx_median: pd.DataFrame, 
                     tx_quantile: pd.DataFrame, endpoint: str, 
                     threshold: float, report_dir: str, **kwargs) -> tuple:
        """Run high-utility itemset mining.
        
        Args:
            **kwargs: Must include 'utility_min_pct' (default 0.01)
            
        Returns:
            tuple: (dict with 'data' DataFrame, list of plot paths)
        """
        self.logger.info(f"Running {self.name}...")
        min_util_pct = kwargs.get('utility_min_pct', 0.01)

        utilities = data[endpoint].clip(lower=0).values
        transactions = BaseDiscretizer()._df_to_transactions(tx_quantile)

        utilities_int = (utilities * 100).astype(int)
        total_utility = np.sum(utilities_int)
        min_utility_threshold = int(total_utility * min_util_pct)

        if total_utility == 0:
            self.logger.warning(
                "HUIM: Skipping, total utility is zero "
                "(endpoint may have all negative values)."
            )
            return {'data': pd.DataFrame()}, []

        # Write transactions with utilities to temporary file for PAMI
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
            _write_utility_transactions_to_file(transactions, utilities_int, tmp_path)

        try:
            # PAMI EFIM expects: EFIM(iFile, minUtil, sep)
            huim = fhm.EFIM(iFile=tmp_path, minUtil=min_utility_threshold, sep='\t')
            huim.mine()  # or startMine()

            # Get patterns as dictionary
            high_utility_itemsets = huim.getPatterns()
        except Exception as e:
            self.logger.error(f"HUIM: EFIM failed. Error: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return {'data': pd.DataFrame()}, []
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not high_utility_itemsets:
            self.logger.info(
                f"HUIM: No high-utility itemsets found with "
                f"min_util_pct={min_util_pct}."
            )
            return {'data': pd.DataFrame()}, []

        # Convert patterns dict to DataFrame
        # PAMI returns patterns as {frozenset: support} or similar
        huim_results = []
        for pattern, utility in high_utility_itemsets.items():
            if isinstance(pattern, (frozenset, set, tuple)):
                itemset_str = ', '.join(str(item) for item in sorted(pattern))
            else:
                itemset_str = str(pattern)
            huim_results.append({'itemset': itemset_str, 'utility': utility})

        huim_df = pd.DataFrame(huim_results)
        huim_df = huim_df.sort_values('utility', ascending=False)

        plot_path = os.path.join(report_dir, f"{endpoint}_huim_plot.png")
        huim_df_top10 = huim_df.head(10).sort_values('utility', ascending=True)
        
        plt.figure(figsize=(10, 7))
        sns.barplot(
            data=huim_df_top10, 
            x='utility', 
            y='itemset', 
            palette='magma'
        )
        plt.title(f'Top 10 High-Utility Itemsets for {endpoint}', fontsize=16)
        plt.xlabel(f'Total Utility (Scaled {endpoint} Score)', fontsize=12)
        plt.ylabel('Feature Pattern', fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return {'data': huim_df}, [plot_path]


class FrequencyAnalyzer(BaseAnalyzer):
    """Run Frequent Itemset Mining (FPGrowth) and generate association rules.
    
    This analyzer discovers common patterns in the data and generates
    association rules showing which feature combinations predict high activity.
    
    Uses the FPGrowth algorithm for efficient frequent pattern mining.
    """

    def __init__(self):
        super().__init__("Frequency Analysis")

    def run_and_plot(self, data: pd.DataFrame, tx_median: pd.DataFrame, 
                     tx_quantile: pd.DataFrame, endpoint: str, 
                     threshold: float, report_dir: str, **kwargs) -> tuple:
        """Run frequent itemset mining and generate association rules.
        
        Args:
            **kwargs: Must include 'freq_min_sup' (default 0.1) and
                'freq_min_conf' (default 0.6)
                
        Returns:
            tuple: (dict with 'data' DataFrame of rules, list of plot paths)
        """
        self.logger.info(f"Running {self.name}...")
        min_sup = kwargs.get('freq_min_sup', 0.1)
        min_conf = kwargs.get('freq_min_conf', 0.6)
        self.min_conf = min_conf  # For reporting
        
        df_tx = tx_median.copy()
        df_tx['Class_Positive'] = (data[endpoint] > threshold)
        df_tx['Class_Negative'] = ~df_tx['Class_Positive']

        transactions = BaseDiscretizer()._df_to_transactions(df_tx)

        # Write transactions to temporary file for PAMI
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_tx:
            tmp_tx_path = tmp_tx.name
            _write_transactions_to_file(transactions, tmp_tx_path, sep='\t')

        # Create temp file for frequent patterns output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_fp:
            tmp_fp_path = tmp_fp.name

        try:
            # Run FPGrowth to find frequent patterns
            # PAMI FPGrowth expects: FPGrowth(iFile, minSup, sep)
            min_sup_count = int(min_sup * len(transactions))
            fim = fpgrowth.FPGrowth(iFile=tmp_tx_path, minSup=min_sup_count, sep='\t')
            fim.mine()  # or startMine()

            # Save frequent patterns to file
            fim.save(tmp_fp_path)

            # Check if any patterns were found
            frequent_itemsets = fim.getPatterns()
            if not frequent_itemsets:
                self.logger.info(
                    f"FIM: No frequent itemsets found with min_sup={min_sup}."
                )
                os.unlink(tmp_tx_path)
                os.unlink(tmp_fp_path)
                return {'data': pd.DataFrame()}, []

            # Generate association rules using confidence
            # PAMI confidence expects: confidence(iFile, minConf, sep)
            ar_miner = post_process.confidence(iFile=tmp_fp_path, minConf=min_conf, sep='\t')
            ar_miner.mine()

            # Get association rules as DataFrame
            rule_df = ar_miner.getAssociationRulesAsDataFrame()

        except Exception as e:
            self.logger.error(f"FIM: FPGrowth/AssociationRules failed. Error: {e}")
            if os.path.exists(tmp_tx_path):
                os.unlink(tmp_tx_path)
            if os.path.exists(tmp_fp_path):
                os.unlink(tmp_fp_path)
            return {'data': pd.DataFrame()}, []
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_tx_path):
                os.unlink(tmp_tx_path)
            if os.path.exists(tmp_fp_path):
                os.unlink(tmp_fp_path)

        if rule_df.empty:
            self.logger.info(
                f"FIM: No association rules found with min_conf={min_conf}."
            )
            return {'data': pd.DataFrame()}, []

        # Filter for rules that predict Class_Positive
        # The column names should be: Antecedent, Consequent, Support, Confidence
        if 'Consequent' in rule_df.columns:
            rules_pos = rule_df[
                rule_df['Consequent'].str.contains('Class_Positive', na=False)
            ]
            if 'Confidence' in rule_df.columns:
                rules_pos = rules_pos.sort_values('Confidence', ascending=False)
        else:
            # If column names are different, use all rules
            rules_pos = rule_df

        # Generate plot if we have rules to visualize
        plot_paths = []
        if not rule_df.empty and len(rule_df) > 1:
            # Normalize column names to lowercase for consistency
            rule_df_plot = rule_df.copy()
            rule_df_plot.columns = [col.lower() for col in rule_df_plot.columns]

            # Filter out Class_Positive and Class_Negative from consequent for general plot
            if 'consequent' in rule_df_plot.columns:
                plot_data = rule_df_plot[
                    ~rule_df_plot['consequent'].astype(str).str.contains(
                        'Class_Positive|Class_Negative',
                        na=False,
                        case=False
                    )
                ]
            else:
                plot_data = rule_df_plot

            if not plot_data.empty:
                plot_path = os.path.join(report_dir, f"{endpoint}_fim_plot.png")
                plt.figure(figsize=(10, 6))

                # Check which columns exist for plotting
                x_col = 'support' if 'support' in plot_data.columns else plot_data.columns[2]
                y_col = 'confidence' if 'confidence' in plot_data.columns else plot_data.columns[3]

                # Use lift if available, otherwise use confidence for hue
                hue_col = 'lift' if 'lift' in plot_data.columns else y_col

                sns.scatterplot(
                    data=plot_data,
                    x=x_col,
                    y=y_col,
                    hue=hue_col,
                    size=hue_col,
                    palette='viridis',
                    sizes=(20, 200),
                    alpha=0.7
                )
                plt.title(
                    f'FIM Association Rules for {endpoint} (min_sup={min_sup})',
                    fontsize=16
                )
                plt.xlabel(x_col.capitalize(), fontsize=12)
                plt.ylabel(y_col.capitalize(), fontsize=12)
                plt.legend(title=hue_col.capitalize(), bbox_to_anchor=(1.05, 1), loc=2)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                plot_paths.append(plot_path)

        return {'data': rules_pos}, plot_paths
