# nbs-predictor
![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![Version](https://img.shields.io/badge/release-v1.2.0-blue.svg)
[![tests](https://github.com/great-lakes-ai-lab/cnbs-predictor/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/great-lakes-ai-lab/cnbs-predictor/actions/workflows/tests.yml)
[![Documentation Status](https://app.readthedocs.org/projects/cnbs-predictor/badge/?version=latest)](https://cnbs-predictor.readthedocs.io/en/latest/)

<p align="center">
  <img src="assets/logo.png" alt="cnbs-logo" width="150"/>
</p>

## Overview
Welcome to nbs-predictor! This software package is designed to forecast key components of **Net Basin Supply (NBS)** for the Laurentian Great Lakes using atmospheric forecast data. Currently, the package uses forecast data from National Oceanic and Atmospheric Administration’s Climate Forecast System (CFS) to predict **precipitation**, **evaporation**, **runoff**, and **net basin supply (NBS)** at monthly intervals up to **12 months** into the future.

### Features
- **Advanced Predictive Algorithms:** Leverages methods like Gaussian Processes and Random Forest for data-driven forecasting.
- **Real-Time Data Processing:** Processes real-time operational forecast data for up-to-date predictions.
- **User-Friendly Interface:** Streamlined setup and interactive notebooks for ease of use.
- **Continuous Improvements:** Actively maintained with regular updates to improve performance, expand functionality, and enhance modeling capabilities.

### Inputs
- **Data Source:** NOAA forecast data from the Climate Forecast System (CFS), which must be downloaded and preprocessed before generating forecasts. Additionally, sea surface temperatures from GLSEA are required as initial conditions. This repository includes notebooks to handle both downloading and preprocessing of both datasets. 
- **Required Datasets:** Please refer to the 'Data Sources' section below for specific files and data organization.

### Targets
Forecast monthly **precipitation (P), evaporation (E), runoff (R), and net basin supply (NBS)** for all of the Laurentian Great Lakes twelve (12) months into the future. By forecasting NBS directly, rather than deriving it solely from the individual component forecasts `(NBS = P − E + R)`, the model minimizes accumulated uncertainty and improves reliability.

### Data Sources

| Abbreviation | Name                                           | Source         | Data Use        |
|--------------|------------------------------------------------|----------------|----------------|
| CFSR         | Climate Forecast System Reanalysis             | NOAA           | Training       |
| CFS          | Climate Forecast System v2                     | NOAA           | Forecasting    |
| GLSEA        | Great Lakes Surface Environmental Analysis     | NOAA GLERL     | Forecasting |
| L2SWBM       | Large Lake Statistical Water Balance Model     | NOAA GLERL     | Training    |
| GLCC         | Great Lakes Coordinating Committee             | USACE / ECCC   | Training |


## Getting Started

### Prerequisites
Before you begin, make sure that the following are installed on your system:

- Conda (Anaconda or Miniconda)
- Python 3.11+

### Installation  

On a **Mac**, begin by opening a **Terminal** window.  
On a **Windows** computer, you will need to open the **Anaconda Prompt**.  
- If you don’t already have Anaconda or Miniconda installed, download and install it from: https://www.anaconda.com/download  

1. **Clone the Repository:** Decide where you want the tool to live on your local machine and clone the repository. In this example, we will add it directly to the Desktop.
    ```bash
    cd Desktop
    ```
    ```bash
    git clone https://github.com/great-lakes-ai-lab/cnbs-predictor.git
    ```
    ```bash
    cd cnbs-predictor
    ```

2. **Set Up the Conda Environment**: Create and activate the Conda environment. 

    ```bash
    conda env create -f requirements/environment.yml
    ```
    ```bash
    conda activate nbs_env
    ```

3. **Set Up Jupyter Kernel**: Register the Conda environment as a Jupyter kernel. This is a one time step. After initial set up, you will not need to run this command again.

    ```bash
    python -m ipykernel install --user --name nbs_env --display-name "Python (nbs_env)"
    ```

### Usage

#### Running Jupyter Lab

After setting up your Conda environment, you can launch **Jupyter Lab** to work with the notebooks. 

```bash
jupyter lab
```

#### Working with Notebooks
1. After starting Jupyter Lab, a new browser window should open.
2. On the left, navigate to the notebooks/production/ directory.
3. First time users who do not have the trained model files (e.g. data/input/models/GP_trained_model_anom.joblib) will need to first run `0_LEF_model_training_anomalies.ipynb`. This script will create and train the scalers, models, and attributes and save them to the appropriate folders. If you already have the required files, skip to step 8.
4. Set your local directory paths in the **User Input** section.
5. Double check the kernel in the top right shows `Python (nbs_env)`. If it does not, click on it to switch kernels. You can also mark "Always start the preferred kernel" to force it to always start with the nbs_env environment.
6. Run the notebook by clicking on the ▶▶ symbol to run all cells or the ▶ to run one cell at a time.
7. You should now have joblib files under models, scalers, and csv/json files under climatology.
8. Open `1_LEF_download_preprocess.ipynb`. Set your local directory paths in the **User Input** section.
9. Run the notebook. This will download CFS data to the specified directory, process the data, and save it to a database `cfs_forecast_data.db`. If you have a database started, you can set **auto** to 'yes' in the notebook and it will automatically pull the last date in the database and pick up where it left off, bringing the database up-to-date. If you are starting a database from scratch, the script will automatically begin downloading CFS data from 9 months prior until today. You can also set **auto** to 'no' and set the specified date range to download and process CFS data. This script takes a while to run, depending on how much data it needs to download and process.
10. Open `2_LEF_forecast_model.ipynb`. Set your local directory paths in the **User Input** section.
11. Run the notebook. This will read in the CFS database, run the data through the saved trained models, and produce a 12 month forecast for **precipitation, evaporation, runoff,** and **NBS**.
12. Script `3_LEF_visualization.ipynb` is an optional notebook you can run to create timeseries plots of the forecast data created in script 2.

## Project Structure

```graphql
cnbs-predictor/
├── assets                                      # Project logos and branding assets
├── CITATION.cff                                # Citation metadata for the project
├── CODE_OF_CONDUCT.md                          # Code of conduct for contributors
├── CONTRIBUTING.md                             # Contribution guidelines
├── data                                        # Directory for storing input and forecast data
│   ├── cfs                                     # Archived pre-processed CFS data
│   ├── cfsr                                    # Archived pre-processed CFSR data used for training
│   ├── forecast                                # Forecast output data
│   ├── glcc                                    # NBS observations from GLCC used for training and validation
│   ├── glsea                                   # GLSEA surface water temperature data
│   ├── input                                   # Model inputs, trained models, scalers, masks, etc.
│   ├── l2swbm                                  # L2SWBM target data for precipitation, evaporation, and runoff
│   ├── probabilities                           # USACE probability files used for CEP calculations
│   └── snodas                                  # SNODAS snow data used for model inputs
├── docs                                        # Project documentation and supporting materials
│   ├── development_history.md                  # Notes on project development and changes
│   ├── experiment_skill_metrics.md             # Skill metric summaries from model experiments
│   ├── experiments.md                          # Experiment descriptions and results
│   ├── Great Lakes NBS Predictor Flyer.pdf     # Project flyer
│   ├── sphinx                                  # Sphinx documentation source files
│   └── wiki-drafts                             # Draft content for GitHub wiki pages
├── images                                      # Team member or project-related images
├── LICENSE                                     # Project license
├── notebooks                                   # Jupyter notebooks
│   ├── exploratory                             # Exploratory and development notebooks
│   ├── production                              # Production-ready notebooks for generating forecasts
│   └── validation                              # Notebooks for model validation
├── README.md                                   # Project overview and setup instructions
├── requirements                                # Conda environment files
│   ├── environment-test.yml                    # Testing environment
│   └── environment.yml                         # Main project environment
├── ROADMAP.md                                  # Project roadmap and planned improvements
├── SECURITY.md                                 # Security policy and reporting guidance
├── src                                         # Source code for data processing, modeling, and utilities
│   ├── data_downloader.py                      # Utilities for downloading input datasets
│   ├── data_loader.py                          # Functions for loading data
│   ├── data_processor.py                       # Data cleaning, transformation, and preprocessing tools
│   ├── database_utils.py                       # Database creation, querying, and management utilities
│   ├── forecast_smoke.py                       # Forecast smoke tests or quick forecast checks
│   ├── hydro_utils.py                          # Hydrologic calculation utilities
│   ├── plotting.py                             # Plotting and visualization functions
│   └── utilities.py                            # General helper functions
└── tests                                       # Test suite
    ├── conftest.py                             # Shared pytest fixtures and configuration
    ├── fixtures                                # Test fixture data
    ├── integration                             # Integration tests across multiple components
    └── unit                                    # Unit tests for individual functions
```

## Contributing
We welcome contributions to cnbs-predictor! Please start a discussion with the maintainers first (@lefitzpatrick, @danijonesocean), preferably in the GitHub Discussions or Issues sections of this repository. Once you have discussed possible changes with the developers:

1. Fork the repository.
2. Clone your fork to your target machine.
3. Create a feature branch for your changes.
4. Make and commit your changes.
5. Push your branch to your fork.
6. Submit a pull request to the main repository.

For detailed contributing instructions, please refer to the [Contributing guidelines](CONTRIBUTING.md). For a summary of current development plans to help map out possible contributions, see our [Roadmap document](ROADMAP.md).

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to follow this code and foster a welcoming and respectful community.

## License

This project is licensed under the [GNU Affero General Public License Version 3.0](LICENSE).

## Acknowledgements

### Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/DaniJonesOcean">
        <img src="./images/dani.jpeg" width="120" alt="Dani"/><br/>
        <b>Dani Jones</b>
      </a>
      <br/>
      Principal Investigator
    </td>
    <td align="center">
      <a href="https://github.com/lefitzpatrick">
        <img src="./images/lindsay.jpg" width="120" alt="Lindsay"/><br/>
        <b>Lindsay Fitzpatrick</b>
      </a>
      <br/>
      Model Developer
    </td>
    <td align="center">
      <a href="https://github.com/DeannaApps">
        <img src="./images/dee.jpeg" width="120" alt="Dee"/><br/>
        <b>Deanna Fielder</b>
      </a>
      <br/>
      Model Testing
    </td>
    <td align="center">
      <a href="https://github.com/jamiewa139">
        <img src="./images/jamie.jpeg" width="120" alt="Jamie"/><br/>
        <b>Jamie Ward</b>
      </a>
      <br/>
      Model Testing
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/pjhopp19">
        <img src="./images/trece.jpeg" width="120" alt="Trece"/><br/>
        <b>Trece Bye</b>
      </a>
      <br/>
      Model Testing
    </td>
    <td align="center">
      <a href="https://github.com/mcanearm">
        <img src="./images/matt.jpeg" width="120" alt="Matt"/><br/>
        <b>Matt McAnear</b>
      </a>
      <br/>
      Contributor
    </td>
    <td align="center">
      <img src="./images/bryan.jpg" width="120" alt="Bryan"/><br/>
      <b>Bryan Mroczka</b>
      <br/>
      Contributor
    </td>
    <td></td> <!-- Empty cell to balance 4 columns -->
  </tr>
</table>



**nbs-predictor** is powered by institutional collaboration from:

<img src="assets/great-lakes-ai-lab-logo.png" alt="Great Lakes AI Lab Logo" height="100" style="padding-right: 10px;">
<img src="assets/CIGLR_LOGO.png" alt="Cooperative Institute for Great Lakes Research (CIGLR) Logo" height="100" style="padding-right: 10px;">
<img src="assets/noaa-glerl-logo.png" alt="NOAA Great Lakes Environmental Research Lab (GLERL) Logo" height="100" style="padding-right: 10px;">
<img src="assets/um-horizontal.png" alt="University of Michigan Logo" width="500" style="padding-right: 10px;">
<img src="assets/usace.png" alt="US Army Corps of Engineers Logo" width="500" style="padding-right: 10px;">

Funding for this project provided by NOAA
