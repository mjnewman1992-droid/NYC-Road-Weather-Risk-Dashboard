# NYC Road Weather Risk Dashboard

## Project Overview

This project is an interactive Dash dashboard that analyzes NYC roadway risk using sample corridor, borough, weather, traffic, and score data.

It includes charts, risk summaries, validation checks, and a basic machine learning model that predicts corridor risk categories.


## Tools Used

- Python
- Dash
- Pandas
- Plotly
- Scikit-learn
- HTML/CSS
- CSV data

## Features

- Interactive dashboard with two tabs: Dashboard and ML Model
- Summary cards for total corridors, high-risk corridors, average score, and top weather issue
- Bar charts, pie chart, and corridor detail cards
- Risk score validation check
- Machine learning prediction check
- Feature importance display

## Machine Learning

The machine learning section uses a Decision Tree Classifier to predict each corridor's risk category.

The model uses score, borough, and weather as input features. Because the current dataset is small, the model is mainly for demonstration and learning purposes.

## Data Source

This project currently uses a sample CSV file created for prototype development. The data includes NYC roadway corridors, boroughs, risk categories, scores, weather conditions, and traffic notes.


## How to Run

1. Open the project folder in VS Code.
2. Open the terminal.
3. Make sure you are inside the nyc-risk-dashboard folder.
4. Run the app with:

```bash
python3 app.py
```

5. Open the local Dash link in a browser.

## Current Limitations

- The dataset is small and manually created.
- The machine learning model is for demonstration only.
- The app does not yet use live traffic or live weather data.
- Predictions may change as more rows are added.

## Future Improvements

- Add more historical corridor records.
- Connect live weather data.
- Connect live traffic data.
- Improve the machine learning model with more training data.
- Add model evaluation charts.
- Deploy the dashboard online.