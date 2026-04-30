# Baby Name Prediction Studio

An end-to-end machine learning project that explores historical U.S. baby-name trends and turns the strongest findings into an interactive prediction website.

## Live App

- Public demo: [babynamesprediction.streamlit.app](https://babynamesprediction.streamlit.app/)
- Current data coverage: `1880-2024`
- Current deployment target: Streamlit Community Cloud on `Python 3.10`

## Project Focus

This project asks a simple but engaging question:

**Can historical naming patterns help predict whether a baby name is likely to become highly popular?**

To answer that, I combined historical U.S. baby-name data with feature engineering, classification modeling, trend analysis, and a Streamlit app that lets users explore names and test prediction scenarios.

## What This Project Includes

- Historical trend exploration using U.S. baby-name records
- Feature engineering based on name structure and popularity patterns
- Logistic regression modeling for Top 100 popularity prediction
- Additional clustering and time-series experimentation in notebooks
- A portfolio-ready Streamlit interface for trend exploration and prediction

## Why This Project Works for a Portfolio

This is more than a notebook-only capstone. It shows an end-to-end workflow:

1. Frame a question that non-technical audiences can understand
2. Clean and reshape a large historical dataset
3. Engineer features for modeling
4. Build a predictive workflow
5. Turn the analysis into a user-facing app

That combination makes it a strong data + product storytelling project.

## Website Experience

The Streamlit app is organized into four sections:

- **Overview**: project framing, dataset size, and recent top-name examples
- **Trend Explorer**: compare names across time using count and ratio-based metrics
- **Prediction Studio**: generate a Top 100 prediction using the trained model
- **Project Insights**: summarize why the project is interesting and how it can improve

The app lives here:

- [streamlit/app1.py](./streamlit/app1.py)

## Data and Features

The main dataset used by the app includes:

- `Name`
- `Year`
- `Gender`
- `Count`
- `Name_Ratio`
- `Gender_Name_Ratio`

The current public app and packaged deployment dataset are updated through `2024`.

The prediction workflow also uses engineered features derived in the modeling notebooks, including:

- `Is_Famous`
- `Gender_Binary`
- `Rolling_Average_Gender_Ratio_5_Years`
- `Vowel_Count`
- `Ends_With_Specified_Letters`

## Modeling Direction

The current interactive prediction experience is centered on a logistic regression model trained to estimate whether a name is likely to land in the Top 100.

Current retrained model snapshot:

- Training window: `1880-2024`
- Training rows: `145,000` top-1000 yearly records
- Test accuracy: `0.9785`
- Test ROC AUC: `0.9953`

Supporting experiments in the repository include:

- clustering analysis
- feature exploration
- time-series forecasting notebooks

For the portfolio version of the project, the app emphasizes the clearest value proposition:

**trend exploration + popularity prediction**

## Repository Structure

- `data_source/` raw and supporting data files
- `models/` trained model artifacts used by the app
- `notebooks/` analysis, feature engineering, and modeling notebooks
- `references/` research and background reading
- `reports_slides/` course reports and presentations
- `src/` supporting project notes
- `streamlit/` interactive app code
- `conda.yml` project environment specification

## Selected Files

- [streamlit/app1.py](./streamlit/app1.py)
- [reports_slides/An analysis of historical data and trends.pdf](./reports_slides/An%20analysis%20of%20historical%20data%20and%20trends.pdf)
- [reports_slides/Machine Learning Models Analysis.pdf](./reports_slides/Machine%20Learning%20Models%20Analysis.pdf)
- [reports_slides/S3_Capstone_BabyName.pdf](./reports_slides/S3_Capstone_BabyName.pdf)

## Running the App

This project includes both a Conda environment file and a pip-friendly requirements file:

- [conda.yml](./conda.yml)
- [requirements.txt](./requirements.txt)

Typical local workflow:

```bash
conda env create -f conda.yml
conda activate capstone
pip install -r requirements.txt
streamlit run streamlit/app1.py
```

## Deploying the App

The easiest way to share this project publicly is through Streamlit Community Cloud.

Current production URL:

- [https://babynamesprediction.streamlit.app/](https://babynamesprediction.streamlit.app/)

Deployment settings:

- Repository root contains `requirements.txt`
- Entrypoint file: `streamlit/app1.py`
- Recommended Python version on Streamlit Cloud: `3.10`

To make deployment practical on GitHub, the app reads a lightweight packaged dataset from:

- `data_source/app_data.pkl.gz`

If that file is not present, the app falls back to the larger local development dataset:

- `notebooks/data.csv`

## Updating to a New SSA Year

If you want to refresh the app with the newest official SSA release:

1. Download the official SSA national data file (`names.zip`) or the yearly file (for example `yob2024.txt`).
2. Save the SSA birth totals page (`numberUSbirths.html`) locally from:
   - `https://www.ssa.gov/OACT/babynames/numberUSbirths.html`
3. Run:

```bash
python3 scripts/update_babyname_data.py \
  --source /path/to/names.zip \
  --birth-totals-html /path/to/numberUSbirths.html \
  --year 2024
```

This updates both:

- `notebooks/data.csv`
- `data_source/app_data.pkl.gz`

If you also want the prediction model to reflect the new year, retrain it after the data refresh:

```bash
python3 scripts/retrain_logistic_model.py
```

That rebuilds:

- `notebooks/Featured_Data.csv`
- `models/preprocessor.pkl`
- `models/logistic_model.pkl`

## Next Improvements

- Add screenshots or GIFs of the app for portfolio readers
- Make the famous-name feature source more transparent
- Link the app from a future portfolio website

## Dataset Note

Some larger datasets and model-related resources are also referenced through this Google Drive folder:

- [Project data folder](https://drive.google.com/drive/folders/1grMuoCSioozmk6MBnLWtH8v2by4OBfEA)

## Author

Ying Zhou
