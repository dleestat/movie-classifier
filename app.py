import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

model, metadata = joblib.load("model/model.joblib"), json.load(open("model/metadata.json"))
classes = metadata["classes"]
tfidfvectorizer = model['columntransformer'].named_transformers_['tfidfvectorizer']
coefficients = pd.DataFrame(
    [estimator.coef_.squeeze() for estimator in model['multioutputclassifier'].estimators_],
    index=classes,
    columns=np.array(tfidfvectorizer.get_feature_names())
)

app = dash.Dash(__name__)
server = app.server
app.title = "Movie Genre Predictor"
app.layout = html.Div([
    html.H1("Movie Genre Predictor"),
    html.Div([
        dcc.Textarea(id="input-text", placeholder="Enter a movie summary here", style=dict(flex=.48, height="254px")),
        html.Figure(id="prediction", style=dict(flex=.52))
    ], style=dict(display="flex")),
    html.Figure(id="interpretation")
])


@app.callback(Output("prediction", "children"), Input("input-text", "value"))
def predict(input_text):
    if not input_text:
        input_text = ""

    tfidf_vector = tfidfvectorizer.transform([input_text]).toarray().squeeze()

    pred = pd.DataFrame({
        "Confidence": np.array(model.predict_proba([[input_text]])).squeeze()[:, 1],
        "Genre": classes
    }).sort_values("Confidence")

    fig = px.bar(pred, x="Confidence", y="Genre", range_x=[0, 1])
    fig.update_layout(
        margin=dict(l=120, r=0, t=0, b=0),
        xaxis=dict(fixedrange=True, tickvals=np.linspace(0, 1, 11)),
        yaxis=dict(fixedrange=True, ticksuffix=" ", title=None)
    )
    fig.update_traces(hovertemplate="%{x:.3f}")
    return dcc.Graph(figure=fig, config=dict(displayModeBar=False), responsive=True, style=dict(height="300px"))


if __name__ == "__main__":
    app.run_server(debug=True)
