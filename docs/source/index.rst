
cnbs-predictor Documentation
=============================

Welcome to the documentation for **cnbs-predictor**! This software package is designed to forecast key components of Net Basin Supply (NBS) for the Laurentian Great Lakes using NOAA's Climate Forecast System (CFS) data. The repository includes features like advanced predictive algorithms, real-time data processing, and easy-to-use Jupyter notebooks for forecasting precipitation, evaporation, and runoff.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
------------
Follow the steps below to set up the cnbs-predictor package.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/great-lakes-ai-lab/cnbs-predictor.git
   ```

2. **Set up the Conda environment:**
   ```bash
   conda env create -f requirements/environment.yml
   conda activate cnbs-predictor-env
   ```

3. **Set up Jupyter Kernel:**
   ```bash
   python -m ipykernel install --user --name cnbs-predictor-env --display-name "Python (cnbs-predictor-env)"
   ```

4. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

Usage
-----
After setting up the environment, you can use the following notebooks to generate forecasts for NBS components:

1. **Forecast Model Training**: Not needed for most users, but useful for model retraining.
2. **Downloading and Preprocessing Input Data**: Prepares NOAA CFS and LS2SWBM data.
3. **Generating Forecasts**: Produces NBS predictions using processed data.

Data Sources
------------
The package utilizes data from NOAA’s Climate Forecast System (CFS) and the Large Lake Statistical Water Balance Model (LS2SWBM). Detailed instructions on downloading and preprocessing these datasets are available in the notebooks.

.. csv-table:: Data Sources
   :header: "Abbreviation", "Name", "Source"
   :widths: 10, 50, 30
   :align: center

   "CFS", "Climate Forecast System v2", "NOAA"
   "LS2SWBM", "Large Lake Statistical Water Balance Model", "NOAA GLERL"

Contributing
------------
We welcome contributions! To contribute, fork the repository and follow these steps:

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Create a feature branch.
4. Make your changes and commit.
5. Push your changes to your fork.
6. Submit a pull request to the main repository.

For more details, check out the [Contributing guidelines](CONTRIBUTING.md) and the [Roadmap document](ROADMAP.md).

License
-------
This project is licensed under the [AGPL-3.0 License](LICENSE).

Acknowledgements
----------------
This project is powered by institutional collaboration from:

- ![Great Lakes AI Lab](assets/great-lakes-ai-lab-logo.png)
- ![CIGLR](assets/CIGLR_LOGO.png)
- ![NOAA GLERL](assets/noaa-glerl-logo.png)
- ![University of Michigan](assets/um-horizontal.png)
- ![US Army Corps of Engineers](assets/usace.png)

Funding provided by NOAA.

API Reference
-------------
The following sections document the key modules and their functions. All classes and functions in these modules are automatically documented using their docstrings.

.. automodule:: src.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.database_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.hydro_utils
   :members:
   :undoc-members:
   :show-inheritance:
