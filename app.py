import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from dash import dash_table
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


app = Dash(__name__)
server = app.server
app.title = "NYC Road Weather Risk Dashboard"

df = pd.read_csv("data/corridors.csv") #Reads corridor data from CSV
df_highways = pd.read_csv("data/nyc_highway_exits.csv")
default_highway = sorted(df_highways["Highway"].unique())[0]
default_direction = sorted(df_highways["Direction"].unique())[0]
df_highway_row_count = len(df_highways)
df_highway_unique_highway_count = df_highways["Highway"].nunique()
df_highway_direction_count = df_highways["Direction"].nunique()
df_sorted = df.sort_values(by='score', ascending=False) #Sorts the Dataframe by score from highest to lowest


def score_category(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Moderate"
    else:
        return "Low" 


df_sorted['calculated_risk'] = df_sorted["score"].apply(score_category)
df_sorted["risk_match"] = df_sorted["risk"] == df_sorted["calculated_risk"]
risk_mismatches = df_sorted[df_sorted["risk_match"] == False]
mismatch_count = len(risk_mismatches)

# Machine learning preparation
X = df_sorted[["score", "borough", "weather"]]
y = df_sorted["risk"]
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.3,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)
model_accuracy_percent = round(model_accuracy * 100)
prediction_results = pd.DataFrame({"Actual Risk": y_test, "Predicted Risk": y_pred})
prediction_results["Correct"] = prediction_results["Actual Risk"] == prediction_results["Predicted Risk"]
prediction_results["Correct"] = prediction_results["Correct"].map({True: "Yes", False: "No"})
prediction_results["Corridor"] = df_sorted.loc[y_test.index, "corridor"].values
prediction_results = prediction_results[["Corridor", "Actual Risk", "Predicted Risk", "Correct"]]
correct_predictions = len(prediction_results[prediction_results["Correct"] == "Yes"])
model_summary = f"The model correctly predicted {correct_predictions} out of {len(y_test)} test corridors with an accuracy of {model_accuracy_percent}%."

feature_importance_df = pd.DataFrame({ "Feature": X_encoded.columns, "Importance": model.feature_importances_})
feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False)
feature_importance_df = feature_importance_df[feature_importance_df["Importance"] > 0]
feature_importance_fig = px.bar(feature_importance_df, x="Importance", y="Feature", orientation="h")
feature_importance_fig.update_layout( title="Model Feature Importance", xaxis_title="Importance", yaxis_title="Feature", plot_bgcolor="white", paper_bgcolor="white", height=300, width=700)


top_corridor = df_sorted.iloc[0]
highest_score = df_sorted['score'].max() 
sorted_corridors = df_sorted.to_dict(orient="records")
total_corridors = len(df_sorted)
high_risk = df_sorted[df_sorted["risk"] == "High"]
high_count = len(high_risk) 
average_risk_score = int(df_sorted['score'].mean().round()) 
most_common_weather_issue = df_sorted["weather"].value_counts().idxmax()
fig = px.bar(df_sorted, x='corridor', y='score', color='risk', color_discrete_map = {"High":"#c62828", "Moderate":"#ef8f00", "Low":"#2e7d32"}, text = "score", hover_data = ["weather", "traffic", 'score'])
fig.update_layout(title="Corridor Risk Scores", legend_title_text = "<u>Risk Category:</u>", xaxis_title = 'Corridor', yaxis_title = 'Risk Score', height=500, plot_bgcolor="white", paper_bgcolor="white", yaxis_range=[0, 110])
fig.update_traces(textposition="outside")
operator_summary = f"{top_corridor['corridor']} has the highest risk score at {highest_score}. {most_common_weather_issue} conditions and traffic delays are the main concern."
validation_summary = f"{mismatch_count} risk label mismatches found after checking scores against the scoring rule."

avg_borough_df = df.groupby('borough')['score'].mean().astype(int).reset_index()


df_avg_fig_sorted_boroughs = avg_borough_df.sort_values(by='score', ascending=False)
df_avg_fig_sorted_boroughs["score_category"] = df_avg_fig_sorted_boroughs["score"].apply(score_category)
avg_fig_sorted_boroughs = px.bar(df_avg_fig_sorted_boroughs, x="borough", y="score", color="score_category", color_discrete_map={"High": "#c62828", "Moderate": "#ef8f00", "Low": "#2e7d32"}, text="score")
avg_fig_sorted_boroughs.update_layout(title="Average Score by Borough", legend_title_text = "<u>Risk Category:</u>", xaxis_title="Borough", yaxis_title="Average Risk Score",height=400, plot_bgcolor="white", paper_bgcolor="white", yaxis_range=[0, 110])
avg_fig_sorted_boroughs.update_traces(textposition="outside")
risk_df = df_sorted['risk'].value_counts().reset_index()
risk_pie = px.pie(risk_df, color='risk', color_discrete_map = {"High":"#c62828", "Moderate":"#ef8f00", "Low":"#2e7d32"}, names='risk', values='count')
risk_pie.update_layout(title="Corridor Risk Distribution", legend_title_text = "<u>Risk Category:</u>", plot_bgcolor="white", paper_bgcolor="white")
risk_pie.update_traces(textinfo='label+percent')



app.layout = html.Div([
    html.Div([
        html.H1("NYC Road Weather Risk Dashboard"),
        html.H2("Combines weather, traffic, and corridor context to estimate NYC roadway risk."),
        html.P("Prototype dashboard using sample NYC corridor data")
    ], className='header'),

    dcc.Tabs([
        dcc.Tab(label="Dashboard", children=[
            html.Div([
                html.H3(f"Corridors: {total_corridors}", className='metric-card metric-total'),
                html.H3(f"High Risk: {high_count}", className='metric-card metric-high'),
                html.H3(f"Avg Score: {average_risk_score}", className='metric-card metric-average'),
                html.H3(f"Top Weather: {most_common_weather_issue}", className='metric-card metric-weather'),
                html.H3(f"Risk Mismatches: {mismatch_count}", className="metric-card")
            ], className="summary-grid"),
            html.H2("Dashboard Charts:", className="section-title"),
            html.Div([
                html.H5("Risk Score Scale"),
                html.H5(["0-39 = ", html.Span("Low", style={"color":"#2e7d32"})]),
                html.H5(["40-69 = ", html.Span("Moderate", style={"color":"#ef8f00"})]),
                html.H5(["70-100 = ", html.Span("High", style={"color":"#c62828"})])
            ], className="card"),
            html.Div([dcc.Graph(figure=fig)], className='card'),
            html.Div([dcc.Graph(figure=avg_fig_sorted_boroughs)], className='card'),
            html.Div([dcc.Graph(figure=risk_pie)], className='card'),
            html.Div([
                html.H2("Current Focus:"),
                html.P(f"{operator_summary}"),
                html.P(f"{validation_summary}")
            ], className="card focus-card"),
            html.H2("Corridor Details:", className='section-title'),
            html.Div([
                html.Div([
                    html.H2(corridor["corridor"]),
                    html.P(f"Score: {corridor['score']}"),
                    html.P(f"Risk: {corridor['risk']}", className=f"risk-{corridor['risk'].lower()}"),
                    html.P(f"Calculated Risk: {corridor['calculated_risk']}"),
                    html.P(f"Borough: {corridor['borough']}"),
                    html.P(f"Weather: {corridor['weather']}"),
                    html.P(f"Traffic: {corridor['traffic']}"),
                ], className=f"corridor-card corridor-{corridor['risk'].lower()}")
                for corridor in sorted_corridors
            ], className='corridor-grid'),
            html.Div([
                html.P("Data Note: Sample corridor data. Risk labels were checked against the scoring rule.")
            ], className="data-note")
        ]),

        dcc.Tab(label="Highway Inventory", children=[
            html.Div([
                html.H2("Highway Inventory:", className="section-title"),
            html.Div([
                html.H3(f"Number of Rows: {df_highway_row_count}"),
                html.H3(f"Number of Highways: {df_highway_unique_highway_count}"),
                html.H3(f"Number of Directions: {df_highway_direction_count}"),
            ], className="summary-grid"),   
        html.H3("Select a Highway:", className="dropdown-title"),
        dcc.Dropdown(
            options=sorted(df_highways["Highway"].unique()),
            value=default_highway,
            id="highway-dropdown"),      
        html.H3("Select a Direction:", className="dropdown-title"),
        dcc.Dropdown(
            options=sorted(df_highways["Direction"].unique()),
            value=default_direction,
            id="direction-dropdown"),
            html.Div([
                dash_table.DataTable(
                id="highway-table",
                data=df_highways[
                    (df_highways["Highway"] == default_highway) & 
                    (df_highways["Direction"] == default_direction)
                ].to_dict('records'),
                columns=[{"name": col, "id": col} for col in df_highways.columns],
                page_size=10,
                style_cell={"textAlign": "left"})
                ])
            ], className='page')]),            
        
        dcc.Tab(label="ML Model", children=[
            html.Div([
                html.H2("Model Prediction Check:", className="ml-title"),
                html.P("This model uses score, borough, and weather to predict the corridor risk category."),
                html.P(f"Model Accuracy: {model_accuracy_percent}%"),
                html.P(f"Training rows: {len(X_train)}"),
                html.P(f"Testing rows: {len(X_test)}"),
                html.P(model_summary),
            dash_table.DataTable(
                data=prediction_results.to_dict("records"),
                columns=[{"name": col, "id": col} for col in prediction_results.columns],
                style_cell={"textAlign": "center"})], className="card"),
        html.H2("Feature Importance:", className="ml-title"),
            html.Div([
                dcc.Graph(figure=feature_importance_fig)], className="card", style={"display": "flex", "justifyContent": "center"}),     
            html.Div([
                html.P("In this prototype, score is the dominant model feature because the current sample dataset is small and risk categories are closely tied to score ranges.")], className="card"),
            html.Div([
                dash_table.DataTable(
                    data=feature_importance_df.to_dict("records"),
                    columns=[{"name": col, "id": col} for col in feature_importance_df.columns],
                    style_cell={"textAlign": "center"})], className="card"),
            html.Div([
                html.P("Model results are for demonstration only until more historical corridor records are added.")], className="card")])])])

@app.callback(
    Output("highway-table", "data"),
    Input("highway-dropdown", "value"),
    Input("direction-dropdown", "value"))


def update_highway_table(selected_highway, selected_direction):
    filtered_df = df_highways[
        (df_highways["Highway"] == selected_highway) &
        (df_highways["Direction"] == selected_direction)]
    return filtered_df.to_dict("records")



import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
