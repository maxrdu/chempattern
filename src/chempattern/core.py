"""Core pipeline orchestration module.

This module provides the main pipeline class that orchestrates
the entire analysis workflow.
"""

import logging
from .data import ChemDataManager
from .discretizers import MedianDiscretizer, QuantileDiscretizer
from .reporting import MarkdownReporter


class ChemPatternPipeline:
    """Orchestrate the loading, preprocessing, analysis, and reporting pipeline.
    
    This class brings together all components of ChemPattern:
    - Data loading and feature generation
    - Feature discretization
    - Running multiple analyzers
    - Report generation
    
    The pipeline processes each endpoint independently, allowing for
    parallel processing in future versions.
    
    Attributes:
        data_manager (ChemDataManager): Handles data loading and features
        analyzers (list): List of analyzer instances to run
        reporter (MarkdownReporter): Generates final reports
        discretizers (dict): Dictionary of fitted discretizers
        
    Example:
        >>> from chempattern import (
        ...     ChemDataManager, ChemPatternPipeline,
        ...     EDAAnalyzer, ContrastAnalyzer,
        ...     MarkdownReporter
        ... )
        >>> 
        >>> data_manager = ChemDataManager(
        ...     main_file='data.csv',
        ...     id_col='ID',
        ...     smiles_col='SMILES'
        ... )
        >>> 
        >>> analyzers = [EDAAnalyzer(), ContrastAnalyzer()]
        >>> reporter = MarkdownReporter('reports')
        >>> 
        >>> pipeline = ChemPatternPipeline(
        ...     data_manager=data_manager,
        ...     analyzers=analyzers,
        ...     reporter=reporter
        ... )
        >>> 
        >>> pipeline.run()
    """
    
    def __init__(self, data_manager: ChemDataManager, 
                 analyzers: list, 
                 reporter: MarkdownReporter):
        """Initialize the pipeline.
        
        Args:
            data_manager (ChemDataManager): Initialized data manager
            analyzers (list): List of analyzer instances (e.g., [EDAAnalyzer(), ...])
            reporter (MarkdownReporter): Initialized reporter
        """
        self.data_manager = data_manager
        self.analyzers = analyzers
        self.reporter = reporter
        self.discretizers = {
            'median': MedianDiscretizer(),
            'quantile': QuantileDiscretizer(num_bins=3)
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, **kwargs):
        """Run the full analysis pipeline for all target endpoints.
        
        For each endpoint:
        1. Load and clean data
        2. Compute statistics and determine threshold
        3. Discretize features (median and quantile)
        4. Run all analyzers
        5. Generate comprehensive report
        
        Args:
            **kwargs: Analysis parameters including:
                - contrast_threshold (float): Activity threshold for contrast analysis.
                    If None, uses median of endpoint.
                - utility_min_pct (float): Minimum utility percentage for HUIM
                    (default: 0.01)
                - freq_min_sup (float): Minimum support for FIM (default: 0.1)
                - freq_min_conf (float): Minimum confidence for association rules
                    (default: 0.6)
                    
        Example:
            >>> pipeline.run(
            ...     contrast_threshold=7.0,
            ...     freq_min_sup=0.2,
            ...     freq_min_conf=0.7
            ... )
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting ChemPattern Pipeline")
        self.logger.info("=" * 60)
        
        endpoints = self.data_manager.get_endpoints_to_analyze()
        self.logger.info(f"Will analyze {len(endpoints)} endpoint(s): {endpoints}")
        
        for idx, endpoint in enumerate(endpoints, 1):
            self.logger.info("")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Analyzing Endpoint {idx}/{len(endpoints)}: {endpoint}")
            self.logger.info(f"{'='*60}")
            
            try:
                data = self.data_manager.get_data_for_endpoint(endpoint)
            except Exception as e:
                self.logger.warning(
                    f"Skipping {endpoint}: Could not load data. Error: {e}"
                )
                continue
                
            if data.empty:
                self.logger.warning(
                    f"Skipping {endpoint}: No valid data after dropping NaNs."
                )
                continue
            
            self.logger.info(f"Analyzing {len(data)} complete records.")
            
            # Get endpoint stats
            stats = {
                'mean': data[endpoint].mean(),
                'median': data[endpoint].median(),
                'std_dev': data[endpoint].std(),
                'count': data[endpoint].count()
            }
            
            self.logger.info(
                f"Endpoint statistics - Mean: {stats['mean']:.4f}, "
                f"Median: {stats['median']:.4f}, StdDev: {stats['std_dev']:.4f}"
            )
            
            # Determine threshold
            threshold = kwargs.get('contrast_threshold')
            if threshold is None:
                threshold = stats['median']
                self.logger.info(
                    f"No threshold specified. Using median: {threshold:.4f}"
                )
            else:
                self.logger.info(f"Using user-specified threshold: {threshold:.4f}")

            # Fit and transform features
            self.logger.info("Discretizing features...")
            features_df = data.drop(columns=[endpoint])
            tx_median = self.discretizers['median'].fit_transform(features_df)
            tx_quantile = self.discretizers['quantile'].fit_transform(features_df)
            self.logger.info(
                f"Created {tx_median.shape[1]} median-discretized features and "
                f"{tx_quantile.shape[1]} quantile-discretized features."
            )
            
            # Run all analyzers
            analysis_results = {}
            for analyzer in self.analyzers:
                self.logger.info(f"Running {analyzer.name}...")
                try:
                    results, plots = analyzer.run_and_plot(
                        data=data,
                        tx_median=tx_median,
                        tx_quantile=tx_quantile,
                        endpoint=endpoint,
                        threshold=threshold,
                        report_dir=self.reporter.report_dir,
                        **kwargs
                    )
                    analysis_results[analyzer.name] = (results, plots)
                    
                    # Log summary
                    if isinstance(results, dict) and 'data' in results:
                        if not results['data'].empty:
                            self.logger.info(
                                f"  -> Found {len(results['data'])} results"
                            )
                        else:
                            self.logger.info("  -> No results found")
                    
                    if plots:
                        self.logger.info(f"  -> Generated {len(plots)} plot(s)")
                        
                except Exception as e:
                    self.logger.error(
                        f"Analyzer '{analyzer.name}' failed for {endpoint}.", 
                        exc_info=True
                    )

            # Generate the final report
            self.logger.info(f"Generating report for {endpoint}...")
            self.reporter.generate(endpoint, stats, analysis_results, **kwargs)
            
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Analysis Complete!")
        self.logger.info("=" * 60)
        self.logger.info(
            f"Reports saved to: {self.reporter.report_dir}"
        )
