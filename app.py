import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import joblib
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import truncate_string

model, metadata = joblib.load("model/model.joblib"), json.load(open("model/metadata.json"))
classes = metadata["classes"]
tfidfvectorizer = model["columntransformer"].named_transformers_["tfidfvectorizer"]
coefficients = pd.DataFrame(
    [estimator.coef_.squeeze() for estimator in model["multioutputclassifier"].estimators_],
    index=classes,
    columns=np.array(tfidfvectorizer.get_feature_names())
)

app = dash.Dash(__name__)
server = app.server
app.title = "Movie Genre Predictor"
app.layout = html.Div([
    html.H1("Movie Genre Predictor"),
    html.H2("Prediction"),
    html.Div([
        dcc.Textarea(id="input-text", placeholder="Enter a movie summary here", style=dict(flex=.48, height="254px")),
        html.Figure(id="prediction", style=dict(flex=.52))
    ], style=dict(display="flex")),
    html.H2("Influential Words"),
    html.Figure(id="interpretation")
])


@app.callback([Output("prediction", "children"), Output("interpretation", "children")], Input("input-text", "value"))
def predict(input_text):
    if not input_text:
        input_text = ""

    predictions = pd.DataFrame({
        "Confidence": np.array(model.predict_proba([[input_text]])).squeeze()[:, 1] * 100,
        "Genre": classes
    }).sort_values("Confidence", ascending=False).reset_index()

    return create_prediction_graph(predictions), create_interpretation_graph(input_text, predictions)


def create_prediction_graph(predictions):
    fig = px.bar(predictions[::-1], x="Confidence", y="Genre", range_x=[0, 100])
    fig.update_layout(
        margin=dict(l=120, r=0, t=0, b=0),
        xaxis=dict(fixedrange=True, ticksuffix="%", tickvals=np.linspace(0, 100, 11)),
        yaxis=dict(fixedrange=True, ticksuffix=" ", title=None)
    )
    fig.update_traces(hovertemplate="%{x:.1f}%")
    return dcc.Graph(figure=fig, config=dict(displayModeBar=False), responsive=True, style=dict(height="300px"))


def create_interpretation_graph(input_text, predictions):
    ordered_classes = list(predictions.Genre)
    tfidf_vector = tfidfvectorizer.transform([input_text]).toarray().squeeze()

    rows = 2
    cols = math.ceil(len(ordered_classes) / rows)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=predictions.apply(lambda x: f"{x.Genre} ({x.Confidence:.0f}%)", 1),
        horizontal_spacing=.02,
        vertical_spacing=.4
    )
    largest_abs_contribution = 0
    for i in range(len(ordered_classes)):
        row, col = i // cols + 1, i % cols + 1

        contributions = coefficients.loc[ordered_classes[i]] * tfidf_vector
        contributions = contributions[contributions != 0]
        contributions = contributions[contributions.abs().nlargest(10).index]
        largest_abs_contribution = max(largest_abs_contribution, contributions.abs().max())

        fig.add_trace(
            go.Bar(
                x=[truncate_string(index, 15) for index in contributions.index],
                y=contributions,
                customdata=contributions.index,
                hovertemplate="%{customdata}: %{y:.3f}<extra></extra>",
                marker_color=["limegreen" if contribution >= 0 else "red" for contribution in contributions],
            ),
            row=row,
            col=col
        )

        if col == 1:
            fig.update_yaxes(title="Contribution", title_font_size=12, row=row, col=col)

    fig.update_annotations(font_size=13)
    fig.update_layout(
        hoverlabel_bordercolor="white",
        margin=dict(l=0, r=0, t=20, b=100),
        showlegend=False
    )
    fig.update_xaxes(
        fixedrange=True,
        showticklabels=bool(np.any(tfidf_vector))
    )
    fig.update_yaxes(
        fixedrange=True,
        range=[-largest_abs_contribution * 1.05, largest_abs_contribution * 1.05],
        showticklabels=False,
        tickvals=[0]
    )

    return dcc.Graph(figure=fig, config=dict(displayModeBar=False), responsive=True, style=dict(height="400px"))


if __name__ == "__main__":
    app.run_server(debug=True)
