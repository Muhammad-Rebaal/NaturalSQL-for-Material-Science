# NaturalSQL for Materials Science

## Project Overview

NaturalSQL for Materials Science is a project that integrates natural language processing capabilities with AiiDA (Automated Interactive Infrastructure and Database for Computational Science) to enable researchers to query computational materials science data using plain English instead of complex SQL queries.

The project leverages AiiDA's powerful provenance tracking and database capabilities to make scientific data more accessible through natural language queries, helping researchers focus on science rather than database query syntax.

## Features

- **Natural Language Queries**: Query your AiiDA database using plain English
- **Automatic SQL Generation**: Converts natural language to optimized SQL queries
- **PDF Report Generation**: Creates comprehensive reports from query results
- **AiiDA Integration**: Works with AiiDA's provenance graph to provide context-aware results
- **Materials Science Focus**: Tailored for computational materials science terminology and workflows

## Repository Structure

```
aiida-core/               # AiiDA core framework (dependency)
nl_query_demo/            # Demo scripts and workflows
  ├── nl_query_workflow.py  # Main workflow for NL queries
  ├── run_workflow.py       # Script to execute workflows
  └── workflow.py           # Workflow definition
nl_query_reports/         # Generated PDF reports
```

## Requirements

- Python 3.8+
- AiiDA 2.5.0+ (2.6.0 recommended)
- PostgreSQL (for production use) or SQLite (for testing)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/natural-sql-materials.git
   cd natural-sql-materials
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install AiiDA**:
   ```bash
   pip install aiida-core
   ```

4. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure AiiDA** (if not already set up):
   ```bash
   verdi setup
   ```

## Configuration

### Setting up AiiDA Profile

If you're starting from scratch, create and configure an AiiDA profile:

```bash
verdi quicksetup  # For quick setup with default values
```

Or for more control:

```bash
verdi setup
```

### Configuring a Computer

Set up a compute resource:

```bash
verdi computer setup -L mycomputer -H localhost -T core.local -S core.direct -w /path/to/work/dir
verdi computer configure core.local mycomputer --safe-interval 0
```

### Setting up Codes

Register computational codes:

```bash
verdi code create core.code.installed --label mycode --computer=mycomputer --default-calc-job-plugin plugin.name --filepath-executable=/path/to/executable
```

## Usage

### Running a Natural Language Query

```bash
cd nl_query_demo
python run_workflow.py --query "Find all calculations that used DFT with a cutoff energy above 400 eV"
```

### Available Demo Queries

The system can handle queries like:

- "Show me all calculations that failed last week"
- "Find structures with more than 50 atoms"
- "List all workflows related to band structure calculations"
- "Count the number of calculations per computer used"
- "What is the average calculation runtime for quantum espresso jobs?"

### Generating Reports

Reports are automatically generated when running queries and saved to the `nl_query_reports` folder with timestamped filenames:

```
nl_query_report_YYYYMMDD_HHMMSS.pdf
```

## Workflow Development

To create your own custom natural language query workflows:

1. Extend the base workflow in `nl_query_workflow.py`
2. Define your specific query patterns in the workflow
3. Register your workflow in `workflow.py`
4. Execute using `nl_query_demo/run_workflow.py`

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project builds upon the [AiiDA framework](https://www.aiida.net), a workflow manager for computational science with a strong focus on provenance, performance, and extensibility.

Please cite the following when using this project:

* S. P. Huber et al., "AiiDA 1.0, a scalable computational infrastructure for automated reproducible workflows and data provenance", Scientific Data 7, 300 (2020); DOI: [10.1038/s41597-020-00638-4](https://doi.org/10.1038/s41597-020-00638-4)

* M. Uhrin et al., "Workflows in AiiDA: Engineering a high-throughput, event-based engine for robust and modular computational workflows", Computational Materials Science 187, 110086 (2021); DOI: [10.1016/j.commatsci.2020.110086](https://doi.org/10.1016/j.commatsci.2020.110086)

## Contact

For questions and support, please open an issue in the GitHub repository or contact the development team at [your-email@example.com](mailto:your-email@example.com).