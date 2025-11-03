"""Report generation module.

This module handles the generation of Markdown reports with
embedded visualizations and data tables.
"""

import os
import logging
from datetime import datetime


class MarkdownReporter:
    """Generate Markdown reports from analyzer results.
    
    The reporter creates comprehensive reports for each endpoint including:
    - Summary statistics
    - Analysis results from all analyzers
    - Embedded visualizations
    - Data tables in Markdown format
    
    Attributes:
        report_dir (str): Directory where reports will be saved
        
    Example:
        >>> reporter = MarkdownReporter(report_dir='my_reports')
        >>> reporter.generate(
        ...     endpoint='IC50',
        ...     stats={'mean': 5.2, 'median': 5.0, 'count': 100},
        ...     analysis_results=results_dict
        ... )
    """
    
    def __init__(self, report_dir: str):
        """Initialize the reporter.
        
        Args:
            report_dir (str): Directory to save reports. Will be created
                if it doesn't exist.
        """
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Reports will be saved to '{self.report_dir}'")

    def generate(self, endpoint: str, stats: dict, 
                 analysis_results: dict, **kwargs):
        """Generate a single Markdown report for one endpoint.
        
        Args:
            endpoint (str): The name of the endpoint
            stats (dict): Basic statistics for the endpoint with keys:
                'count', 'mean', 'median', 'std_dev'
            analysis_results (dict): Dictionary mapping analyzer names to
                (results, plots) tuples
            **kwargs: Additional parameters to include in report context
                (e.g., freq_min_conf for frequency analysis)
        """
        report_path = os.path.join(self.report_dir, f"Report_{endpoint}.md")
        self.logger.info(f"Generating report: {report_path}")
        
        report = f"# Pattern Analysis Report for: {endpoint}\n\n"
        report += f"* **Analysis Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"* **Total Molecules Analyzed:** {stats['count']}\n"
        report += f"* **Endpoint Mean:** {stats['mean']:.4f}\n"
        report += f"* **Endpoint Median:** {stats['median']:.4f}\n\n"
        report += "---\n"

        # Loop through analyzers in a specific order for a clean report
        analyzer_order = [
            "Exploratory Data Analysis", 
            "Contrast Analysis", 
            "High-Utility Analysis",
            "Frequency Analysis"
        ]

        for name in analyzer_order:
            if name not in analysis_results:
                continue
            
            results, plots = analysis_results[name]
            report += f"## {name}\n\n"
            
            # --- Handle EDA ---
            if name == "Exploratory Data Analysis":
                report += (
                    "*High-level summary of the endpoint and "
                    "feature correlations.*\n\n"
                )
                if plots:
                    report += f"![Endpoint Distribution]({os.path.basename(plots[0])})\n\n"
                    report += f"![Feature Correlation]({os.path.basename(plots[1])})\n\n"
            
            # --- Handle Contrast Analysis ---
            if name == "Contrast Analysis":
                threshold = results.get('threshold', stats['median'])
                report += (
                    f"*These patterns are significantly more common in "
                    f"high-activity molecules (`{endpoint}` > {threshold:.4f}) "
                    f"than low-activity ones.*\n\n"
                )
                if 'data' in results and not results['data'].empty:
                    report += (
                        "**Top 10 Emerging Patterns "
                        "(Correlated with HIGH Activity):**\n"
                    )
                    report += (
                        results['data'].head(10).to_markdown(
                            index=False, 
                            floatfmt=".4f"
                        ) + "\n\n"
                    )
                    if plots:
                        report += f"![CEP Plot]({os.path.basename(plots[0])})\n\n"
                else:
                    report += "*No significant emerging patterns were found.*\n\n"
            
            # --- Handle Utility Analysis ---
            if name == "High-Utility Analysis":
                report += (
                    "*These patterns contribute the most to the total score "
                    "across the entire dataset. A high-utility pattern is a "
                    "strong driver of potency.*\n\n"
                )
                if 'data' in results and not results['data'].empty:
                    report += "**Top 10 High-Utility Patterns:**\n"
                    report += (
                        results['data'].head(10).to_markdown(index=False) + "\n\n"
                    )
                    if plots:
                        report += f"![HUIM Plot]({os.path.basename(plots[0])})\n\n"
                else:
                    report += "*No high-utility patterns were found.*\n\n"

            # --- Handle Frequency Analysis ---
            if name == "Frequency Analysis":
                # Get threshold from contrast analysis results for consistency
                threshold = stats['median']
                if "Contrast Analysis" in analysis_results:
                    threshold = analysis_results["Contrast Analysis"][0].get(
                        'threshold', 
                        stats['median']
                    )
                
                report += (
                    f"*These are the most common and reliable \"if-then\" rules "
                    f"found in your data (for `Class_Positive` = `{endpoint}` "
                    f"> {threshold:.4f})*.\n\n"
                )
                
                if 'data' in results and not results['data'].empty:
                    min_conf = kwargs.get('freq_min_conf', 0.6)
                    report += (
                        f"**Most Confident Rules for HIGH Activity "
                        f"(min_conf > {min_conf}):**\n"
                    )
                    report += (
                        results['data'].head(10).to_markdown(
                            index=False, 
                            floatfmt=".4f"
                        ) + "\n\n"
                    )
                    if plots:
                        report += f"![FIM Plot]({os.path.basename(plots[0])})\n\n"
                else:
                    report += "*No frequent rules were found.*\n\n"
            
            report += "---\n"

        try:
            with open(report_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved successfully: {report_path}")
        except Exception as e:
            self.logger.error(f"Error writing report file: {e}")
