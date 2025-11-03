"""Command-line interface for ChemPatternTool.

This module provides the CLI for running ChemPattern analyses
from the command line.
"""

import argparse
import logging

from .data import ChemDataManager
from .analyzers import (
    EDAAnalyzer, 
    ContrastAnalyzer, 
    UtilityAnalyzer, 
    FrequencyAnalyzer
)
from .reporting import MarkdownReporter
from .core import ChemPatternPipeline


def main():
    """Main function to run the tool from the command line.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="ChemPatternTool: Composable Chemical Pattern Mining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with automatic feature generation
  chempattern --main_file data.csv --id_col ID --smiles_col SMILES

  # With pre-computed features
  chempattern --main_file data.csv --id_col ID --smiles_col SMILES \\
              --feature_file features.csv

  # Target specific endpoints
  chempattern --main_file data.csv --id_col ID --smiles_col SMILES \\
              --target_endpoints IC50 EC50

  # Advanced configuration
  chempattern --main_file data.csv --id_col ID --smiles_col SMILES \\
              --contrast_threshold 7.0 --freq_min_sup 0.2 \\
              --freq_min_conf 0.7 --log_level DEBUG

For more information, visit: https://github.com/maxrdu/chempattern
        """
    )
    
    # ===== Data Input Arguments =====
    data_group = parser.add_argument_group('Data Input')
    data_group.add_argument(
        '--main_file', 
        type=str, 
        required=True,
        help='Path to main CSV file with SMILES and endpoints (required)'
    )
    data_group.add_argument(
        '--id_col', 
        type=str, 
        required=True,
        help='Name of the ID column (required)'
    )
    data_group.add_argument(
        '--smiles_col', 
        type=str, 
        required=True,
        help='Name of the SMILES column (required)'
    )
    data_group.add_argument(
        '--feature_file', 
        type=str, 
        default=None,
        help='Optional path to pre-computed feature CSV file. '
             'If not provided, RDKit descriptors will be generated.'
    )
    data_group.add_argument(
        '--target_endpoints', 
        type=str, 
        nargs='+', 
        default=None,
        help='Specific endpoint columns to analyze. '
             'If not provided, all numeric columns will be analyzed. '
             'Example: --target_endpoints IC50 EC50'
    )
    
    # ===== Analysis Configuration Arguments =====
    analysis_group = parser.add_argument_group('Analysis Configuration')
    analysis_group.add_argument(
        '--report_dir', 
        type=str, 
        default='ChemPattern_Reports',
        help='Directory for output reports and plots (default: ChemPattern_Reports)'
    )
    analysis_group.add_argument(
        '--contrast_threshold', 
        type=float, 
        default=None,
        help='Activity threshold for contrast analysis. '
             'Molecules above this value are considered "high activity". '
             'If not specified, the median will be used.'
    )
    analysis_group.add_argument(
        '--utility_min_pct', 
        type=float, 
        default=0.01,
        help='Minimum utility percentage for high-utility itemset mining '
             '(default: 0.01 = 1%% of total utility)'
    )
    analysis_group.add_argument(
        '--freq_min_sup', 
        type=float, 
        default=0.1,
        help='Minimum support threshold for frequent itemset mining '
             '(default: 0.1 = 10%% of transactions)'
    )
    analysis_group.add_argument(
        '--freq_min_conf', 
        type=float, 
        default=0.6,
        help='Minimum confidence threshold for association rules '
             '(default: 0.6 = 60%% confidence)'
    )
    
    # ===== Logging Argument =====
    parser.add_argument(
        '--log_level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity level (default: INFO)'
    )
    
    # ===== Version =====
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("ChemPattern v0.1.0 - Composable Chemical Pattern Mining")
    logger.info("")

    try:
        # 1. Initialize Data Manager
        logger.info("Step 1/3: Loading data...")
        data_manager = ChemDataManager(
            main_file=args.main_file,
            id_col=args.id_col,
            smiles_col=args.smiles_col,
            feature_file=args.feature_file,
            target_endpoints=args.target_endpoints
        )
        
        # 2. Configure Analyzers
        logger.info("Step 2/3: Configuring analyzers...")
        # This is the key to extensibility:
        # An experienced user can add their own `MyCustomAnalyzer()` here.
        analyzers = [
            EDAAnalyzer(),
            ContrastAnalyzer(),
            UtilityAnalyzer(),
            FrequencyAnalyzer()
        ]
        logger.info(f"Enabled analyzers: {[a.name for a in analyzers]}")
        
        # 3. Initialize Reporter
        reporter = MarkdownReporter(report_dir=args.report_dir)

        # 4. Create and run pipeline
        logger.info("Step 3/3: Running analysis pipeline...")
        logger.info("")
        pipeline = ChemPatternPipeline(
            data_manager=data_manager,
            analyzers=analyzers,
            reporter=reporter
        )
        
        pipeline.run(
            contrast_threshold=args.contrast_threshold,
            utility_min_pct=args.utility_min_pct,
            freq_min_sup=args.freq_min_sup,
            freq_min_conf=args.freq_min_conf
        )
        
        logger.info("")
        logger.info("✓ Analysis completed successfully!")
        logger.info(f"✓ Results saved to: {args.report_dir}")
        return 0

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error("A critical error occurred:")
        logger.error(str(e))
        logger.error("=" * 60)
        logger.error("")
        logger.error("For help, run: chempattern --help")
        logger.error("", exc_info=args.log_level == 'DEBUG')
        return 1


if __name__ == "__main__":
    exit(main())
