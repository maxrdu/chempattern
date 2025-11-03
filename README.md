# ChemPattern

**Composable Chemical Pattern Mining Toolkit for Drug Discovery**

ChemPattern is a flexible, extensible Python toolkit for discovering patterns in chemical datasets using multiple data mining techniques including frequent itemset mining, contrast pattern mining, and high-utility itemset mining.

## Features

- 🧪 **Automatic RDKit Feature Generation**: Compute molecular descriptors automatically from SMILES
- 🔍 **Multiple Mining Algorithms**: Frequent patterns, emerging patterns, high-utility itemsets
- 📊 **Exploratory Data Analysis**: Correlation heatmaps and distribution plots
- 🎯 **Composable Architecture**: Easily add custom analyzers
- 📝 **Automated Reporting**: Generates comprehensive Markdown reports with visualizations
- 🔧 **Flexible Configuration**: Command-line interface with sensible defaults

## Installation

### From Source

```bash
git clone https://github.com/yourusername/chempattern.git
cd chempattern
pip install -e .
```

### Dependencies

The main dependencies are:
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- rdkit (cheminformatics)
- pami (pattern mining algorithms)
- scikit-learn (preprocessing)

**Note on RDKit**: RDKit can be tricky to install. We recommend using conda:

```bash
conda create -n chempattern python=3.10
conda activate chempattern
conda install -c conda-forge rdkit
pip install -e .
```

## Quick Start

### Command Line Usage

The simplest usage requires a CSV file with molecule IDs and SMILES:

```bash
chempattern \
    --main_file mydata.csv \
    --id_col compound_id \
    --smiles_col smiles
```

This will:
1. Generate RDKit molecular descriptors
2. Detect all numeric columns as endpoints
3. Run all analysis types
4. Generate reports in `ChemPattern_Reports/`

### With Pre-computed Features

If you already have computed features:

```bash
chempattern \
    --main_file endpoints.csv \
    --id_col compound_id \
    --smiles_col smiles \
    --feature_file precomputed_features.csv
```

### Targeting Specific Endpoints

```bash
chempattern \
    --main_file data.csv \
    --id_col mol_id \
    --smiles_col smiles \
    --target_endpoints IC50 EC50 Solubility
```

### Advanced Configuration

```bash
chempattern \
    --main_file data.csv \
    --id_col mol_id \
    --smiles_col smiles \
    --report_dir my_analysis \
    --contrast_threshold 7.0 \
    --freq_min_sup 0.2 \
    --freq_min_conf 0.7 \
    --utility_min_pct 0.05 \
    --log_level DEBUG
```

## Python API Usage

You can also use ChemPattern programmatically:

```python
from chempattern import (
    ChemDataManager,
    ChemPatternPipeline,
    EDAAnalyzer,
    ContrastAnalyzer,
    UtilityAnalyzer,
    FrequencyAnalyzer,
    MarkdownReporter
)

# Load data
data_manager = ChemDataManager(
    main_file='data.csv',
    id_col='compound_id',
    smiles_col='smiles',
    target_endpoints=['IC50', 'Solubility']
)

# Configure analyzers
analyzers = [
    EDAAnalyzer(),
    ContrastAnalyzer(),
    UtilityAnalyzer(),
    FrequencyAnalyzer()
]

# Setup reporting
reporter = MarkdownReporter(report_dir='results')

# Run pipeline
pipeline = ChemPatternPipeline(
    data_manager=data_manager,
    analyzers=analyzers,
    reporter=reporter
)

pipeline.run(
    contrast_threshold=7.0,
    freq_min_sup=0.15,
    freq_min_conf=0.6
)
```

## Extending ChemPattern

### Adding a Custom Analyzer

```python
from chempattern.analyzers import BaseAnalyzer
import pandas as pd

class MyCustomAnalyzer(BaseAnalyzer):
    """Your custom analysis."""
    
    def __init__(self):
        super().__init__("My Custom Analysis")
    
    def run_and_plot(self, data, tx_median, tx_quantile, 
                     endpoint, threshold, report_dir, **kwargs):
        # Your analysis logic here
        results = {'data': pd.DataFrame()}
        plot_paths = []
        
        return results, plot_paths

# Use it in your pipeline
analyzers = [
    EDAAnalyzer(),
    MyCustomAnalyzer(),  # Your custom analyzer
    ContrastAnalyzer()
]
```

## Output

ChemPattern generates:

1. **Markdown Reports**: One per endpoint with:
   - Summary statistics
   - Analysis results from all enabled analyzers
   - Interpretations and actionable insights

2. **Visualizations**:
   - Distribution plots
   - Correlation heatmaps
   - Bar charts for top patterns
   - Scatter plots for association rules

3. **Data Tables**: Embedded in Markdown reports with:
   - Emerging patterns (contrast analysis)
   - High-utility itemsets
   - Association rules

## Analysis Types

### 1. Exploratory Data Analysis (EDA)
- Distribution plots for endpoints
- Feature correlation heatmaps
- Summary statistics

### 2. Contrast Pattern Mining
- Identifies patterns more common in high-activity vs low-activity molecules
- Uses ERMiner algorithm
- Reports growth rates

### 3. High-Utility Itemset Mining
- Finds patterns that contribute most to total activity
- Uses FHM algorithm
- Weighted by endpoint values

### 4. Frequent Pattern Mining
- Discovers common co-occurring features
- Generates association rules
- Reports support, confidence, lift

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--main_file` | Path to main CSV file (required) | - |
| `--id_col` | Name of ID column (required) | - |
| `--smiles_col` | Name of SMILES column (required) | - |
| `--feature_file` | Pre-computed features CSV | None |
| `--target_endpoints` | Specific endpoints to analyze | All numeric |
| `--report_dir` | Output directory | `ChemPattern_Reports` |
| `--contrast_threshold` | Activity threshold for contrast | Median |
| `--utility_min_pct` | Min utility % for HUIM | 0.01 |
| `--freq_min_sup` | Min support for FIM | 0.1 |
| `--freq_min_conf` | Min confidence for rules | 0.6 |
| `--log_level` | Logging verbosity | INFO |

## Input File Format

### Main File (Required)
CSV with at least:
- An ID column (any name)
- A SMILES column (any name)
- One or more numeric endpoint columns

Example:
```csv
compound_id,smiles,IC50,Solubility
CHEM001,CCO,5.2,2.1
CHEM002,c1ccccc1,7.8,1.5
```

### Feature File (Optional)
CSV with:
- The same ID column
- Numeric feature columns

Example:
```csv
compound_id,MolWt,LogP,TPSA
CHEM001,46.07,-0.3,20.2
CHEM002,78.11,2.1,0.0
```

## Troubleshooting

### RDKit Installation Issues

If you get RDKit import errors:
```bash
# Use conda for RDKit
conda install -c conda-forge rdkit
```

### Memory Issues with Large Datasets

For datasets with >10,000 compounds:
- Increase minimum support thresholds
- Reduce utility minimum percentage
- Process endpoints separately

### No Patterns Found

If analyses return no results:
- Lower the minimum support (`--freq_min_sup`)
- Lower the minimum confidence (`--freq_min_conf`)
- Check for sufficient variation in your data
- Verify your threshold is appropriate

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use ChemPattern in your research, please cite:

```bibtex
@software{chempattern,
  title = {ChemPattern: Composable Chemical Pattern Mining},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/chempattern}
}
```

## Support

- 📧 Email: your.email@example.com
- 🐛 Issues: https://github.com/yourusername/chempattern/issues
- 📖 Documentation: https://chempattern.readthedocs.io
