
# 💻 Laptop Price Predictor

A machine learning project to predict laptop prices based on specifications and features.

## About the Project

Core focus:

- Building machine learning pipelines for laptop price prediction 
- Documenting the pipeline development process through Jupyter notebook 
- Providing an interactive Streamlit app for demonstrating the trained pipelines

## 📂 Quick Access

## 📂 Quick Access

- [Jupyter Notebook](./Laptop_Price_Predictor.ipynb) — analysis and pipeline development
- [Home.py](./Home.py) — main Streamlit app source code
- [Deployed Streamlit App](https://laptoppricepredictor-unipi.streamlit.app/) — interactive web app
- [Machine Learning Pipelines](./pipes/) — folder containing the trained pipelines
- [Google Drive Folder](https://drive.google.com/drive/folders/1KzkWtlL4KB8dkorFx4SIqQwyxXpf4QDI?usp=drive_link) — additional project materials

## 📂 Project Structure

├── Home.py
├── Laptop_Price_Predictor.ipynb
├── csv/                          # CSV datasets
│   ├── laptop_data.csv
│   ├── paper_performance.csv
│   ├── performance_obtained.csv
│   ├── preprocesseddata.csv
│   ├── rmse.csv
├── pages/                        # Streamlit pages
│   ├── Data.py
│   ├── Results.py
├── pipes/                        # Machine learning pipelines
│   ├── pipe_dt.pkl
│   ├── pipe_knn.pkl
│   ├── pipe_lasso.pkl
│   ├── pipe_lr.pkl
│   ├── pipe_rf.pkl
│   ├── pipe_ridge.pkl
├── pipe.pkl                      # Final trained pipeline
├── docs/                         # Documentation and presentation
│   ├── presentation.pdf
│   ├── report.pdf
│   ├── report/                   # LaTeX source files for the report
│   │   ├── tex.zip
│   │   └── README.md 
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore rules
├── icon.png                      # Project icon
└── README.md                     # Main project documentation

## 🚀 Installation

1️⃣ Clone the repo:

    git clone https://github.com/martinaspeciale/laptop-price-predictor.git
    cd laptop-price-predictor

2️⃣ Create and activate virtual environment (optional but recommended):

    python3 -m venv venv
    source venv/bin/activate

3️⃣ Install dependencies:

    pip install -r requirements.txt

## 🏃‍♂️ Usage

### Run the Streamlit app:

```bash
streamlit run Home.py
```

### Run the Jupyter notebook:

```bash
jupyter notebook Laptop_Price_Predictor.ipynb
```

## 📊 Data

The project uses preprocessed datasets provided in the `csv/` folder.

## 🎁 Pre-trained models

The machine learning pipelines provided in the pipes/ folder are built and trained as part of this project. They enable fast and reproducible laptop price predictions.

## 📄 Documentation

Presentations and reports are organized in the `docs/` folder.

