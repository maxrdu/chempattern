"""ChemPattern: Composable chemical pattern mining toolkit for drug discovery.

This package provides a flexible framework for discovering patterns in chemical
datasets using multiple data mining techniques including frequent itemset mining,
contrast pattern mining, and high-utility itemset mining.

Example:
    >>> from chempattern import ChemDataManager, ChemPatternPipeline
    >>> from chempattern import EDAAnalyzer, ContrastAnalyzer
    >>> 
    >>> data_manager = ChemDataManager(
    ...     main_file='data.csv',
    ...     id_col='compound_id',
    ...     smiles_col='smiles'
    ... )
    >>> 
    >>> analyzers = [EDAAnalyzer(), ContrastAnalyzer()]
    >>> pipeline = ChemPatternPipeline(data_manager, analyzers, reporter)
    >>> pipeline.run()
"""

from .data import ChemDataManager
from .discretizers import BaseDiscretizer, MedianDiscretizer, QuantileDiscretizer
from .analyzers import (
    BaseAnalyzer,
    EDAAnalyzer,
    ContrastAnalyzer,
    UtilityAnalyzer,
    FrequencyAnalyzer,
)
from .reporting import MarkdownReporter
from .core import ChemPatternPipeline

__version__ = "0.1.0"
__author__ = "Max Rausch-Dupont"
__email__ = "max@rausch-dupont.de"

__all__ = [
    # Core
    "ChemPatternPipeline",
    # Data Management
    "ChemDataManager",
    # Discretizers
    "BaseDiscretizer",
    "MedianDiscretizer",
    "QuantileDiscretizer",
    # Analyzers
    "BaseAnalyzer",
    "EDAAnalyzer",
    "ContrastAnalyzer",
    "UtilityAnalyzer",
    "FrequencyAnalyzer",
    # Reporting
    "MarkdownReporter",
]
