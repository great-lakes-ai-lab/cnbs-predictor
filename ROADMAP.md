# Project Roadmap: Great Lakes NBS Predictor

This document outlines the high-level release schedule and the prioritized CI/CD implementation plan for the `nbs-predictor` project.

## Summary Schematic

<img width="1559" height="882" alt="four-stage-schematic-2025-10-02" src="https://github.com/user-attachments/assets/06d804cf-a5eb-409b-8f53-ea2321e325e1" />

Underlined terms indicate new planned features for each release

## CI/CD Roadmap

| Milestone | Target Date | Description | Status |
| :---: | :---: | :--- | :--- |
| Phase 1: Pipeline Reliability | December 2025 | Complete and automate all P1 (Data Integrity) and P2 (Code Stability) tests. Establish a stable, reliable data ingestion and feature engineering pipeline. | (TODO) |
| Phase 2: Final MLOps Release | June 2026 | Complete and integrate all P3 (Model Quality) tests. Finalize the ensemble model selection, performance baselines, and deploy the official production forecast. | (TODO) |

## CI/CD Test Implementation Roadmap

The following tests are prioritized from P1 (most critical) to P3 (MLOps-specific) and must be completed sequentially to ensure pipeline stability before moving to model quality.

### 1. P1: Data Integrity and Input Pipeline Tests (CRITICAL)

These tests must execute first, validating the raw inputs before processing.

| Test Category | Specific Test | Rationale | Status | GH Link |
| :--- | :--- | :--- | :--- | :--- |
| External Source Health | Connection/Download Smoke Test | Verify connectivity to the primary AWS and backup NOAA NCEI CFS endpoints, as well as sources of other data such as GLSEA. | (TODO) | \[Issue/PR #\] |
| Raw Schema Validation | Input Data Schema Check | Ensure the newly downloaded data matches the expected format required by `data_processing.py`. | (TODO) | \[Issue/PR #\] |
| Time Series Constraints | Monotonicity & Completeness | Verify that the timestamp column is strictly increasing (monotonic) and check for large, unexpected gaps. | (TODO) | \[Issue/PR #\] |
| Data Quality Check | Missing Value Threshold | Assert that the percentage of `NULL` or missing values in critical columns is below a predefined threshold. | (TODO) | \[Issue/PR #\] |
| Time Horizon Check | Forecast Lookahead Validation | Assert that the raw CFS data contains valid forecast values for the full required time horizon (up to 9 months ahead). | (TODO) | \[Issue/PR #\] |

### 2. P2: Code and Environment Stability Tests (HIGH)

These ensure the core software environment and utility functions are stable.

| Test Category | Specific Test | Rationale | Status | GH Link |
| :--- | :--- | :--- | :--- | :--- |
| Environment Check | Dependency & Setup Test | Confirms that all dependencies install correctly in the CI environment. | (TODO) | \[Issue/PR #\] |
| Unit Testing | Core `src/` Unit Tests | Execute all unit tests for utility files (`hydro_utils.py`, etc.) to ensure feature engineering and data transformations are mathematically correct. | (TODO) | \[Issue/PR #\] |
| Notebook Smoke Test | Production Notebook Execution | Assert the minimal production notebook runs to completion without raising an exception. | (TODO) | \[Issue/PR #\] |
| Configuration Loading | Path and Credential Check | Verify that the code can successfully load configuration, environment variables, or file paths required for access. | (TODO) | \[Issue/PR #\] |
| Internal Database Utility Tests | Schema and Integrity Check | Verify that the database table exists and all 7 expected columns have the correct data types. | (TODO) | \[Issue/PR #\] |
| | Primary Key Constraint Check | Assert the database correctly handles `INSERT OR REPLACE` logic to prevent duplicate records. | (TODO) | \[Issue/PR #\] |
| | CRUD Read/Write Smoke Test | Insert a known dummy record and immediately retrieve it to confirm basic database interaction. | (TODO) | \[Issue/PR #\] |
| | Next Run Date Logic Test | Test `get_next_run()` with mocked entries to assert correct calculation of the expected next midnight start date. | (TODO) | \[Issue/PR #\] |
| | DataFrame Loading Test | Test `load()` to ensure it consistently returns a valid, non-empty Pandas DataFrame. | (TODO) | \[Issue/PR #\] |

### 3. P3: Model and Output Quality Tests (HIGH / MLOps Specific)

These validate the behavior of the complex machine learning components and the final forecast output.

| Test Category | Specific Test | Rationale | Status | GH Link |
| :--- | :--- | :--- | :--- | :--- |
| Model Integrity | Model Loading & Output Format | Verify that all individual ML models can be loaded successfully and adhere to the correct data structure, shape, and type. | (TODO) | \[Issue/PR #\] |
| Ensemble Logic Test | Weighted Calculation Check | Assert the final weighted NBS forecast is calculated correctly using known inputs. | (TODO) | \[Issue/PR #\] |
| Model Performance | Baseline Regression Test | Fail the pipeline if the core metric (e.g., RMSE on NBS) degrades more than a set tolerance (e.g., > 1%) compared to the production model. | (TODO) | \[Issue/PR #\] |
| Hydrological Sanity | Forecast Boundary Check | Assert that the final NBS forecast for the next 12 months falls within historically plausible boundaries ($X_{min}$ and $X_{max}$). | (TODO) | \[Issue/PR #\] |
| GP Uncertainty Check | Prediction Confidence Test | Ensure the Gaussian Process model's output for the measure of uncertainty (standard deviation or variance) is non-zero. | (TODO) | \[Issue/PR #\] |
