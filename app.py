import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
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
from src.utils import remove_leading_article, truncate_string

model = joblib.load("model/model.joblib")
metadata = json.load(open("model/metadata.json"))
example_inputs = json.load(open("assets/example_inputs.json"))
classes = metadata["classes"]
tfidfvectorizer = model["columntransformer"].named_transformers_["tfidfvectorizer"]
coefficients = pd.DataFrame(
    [estimator.coef_.squeeze() for estimator in model["multioutputclassifier"].estimators_],
    index=classes,
    columns=np.array(tfidfvectorizer.get_feature_names())
)

app = dash.Dash(
    __name__,
    title="Real-Time Movie Classifier",
    update_title=None,
    external_stylesheets=[dbc.themes.MATERIA]
)
server = app.server
app.layout = html.Div([
    html.Div([
        html.H1("Real-Time Movie Classifier"),
        html.P("Classify a movie and interpret the prediction in real-time.", style=dict(marginTop="-.5em")),
        dbc.Button(
            [
                html.Img(src="assets/GitHub-Mark-120px-plus.png", height="15em", style=dict(marginRight=".5em")),
                "View on GitHub"
            ],
            href="https://github.com/dleestat/movie-classifier",
            size="sm",
            target="_blank",
            style=dict(marginTop="-.4em")
        )
    ], style=dict(textAlign="center")),
    html.H2("Prediction"),
    html.P([
        "The following classifier is trained on the ",
        html.A("CMU Movie Summary Corpus", href="http://www.cs.cmu.edu/~ark/personas/"),
        ". We use binary relevance with logistic regression using TF-IDF text features."
    ]),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="example-input",
                clearable=True,
                options=[dict(label=i, value=i) for i in sorted(example_inputs.keys(), key=remove_leading_article)],
                optionHeight=25,
                placeholder="Select an example",
                searchable=False,
                style=dict(fontSize="13px", marginBottom="6px", width="360px"),
            ),
            dcc.Textarea(
                id="input-text",
                placeholder="Enter a movie summary",
                style=dict(border="1px solid #ccc", borderRadius="4px", fontSize="13px", height="210px",
                           padding="9px 10px", resize="none", width="100%")
            )
        ], style=dict(flex=.48)),
        html.Figure(id="prediction", style=dict(flex=.52, margin="0px 0px 0px 70px"))
    ], style=dict(display="flex")),
    html.H2("Influential Words"),
    html.P([
        "To identify words that heavily contribute to the above prediction, we define a word's ",
        html.Em("contribution"),
        " to a genre's prediction as the product of:"
    ], style=dict(marginBottom="0")),
    html.Ol([
        html.Li("The model coefficient corresponding to the word and genre."),
        html.Li("The input's TF-IDF value corresponding to the word.")
    ], style=dict(marginTop=".4em", marginBottom="0")),
    html.P("For each genre, we display the words with the largest contributions below.", style=dict(marginTop=".4em")),
    html.Figure(id="interpretation")
], style=dict(margin="auto", marginTop="1em", width="90%"))


@app.callback([Output("input-text", "value"), Output("input-text", "disabled")], Input("example-input", "value"))
def update_example_input(example_input):
    return (example_inputs[example_input], True) if example_input else ("", False)


@app.callback([Output("prediction", "children"), Output("interpretation", "children")], Input("input-text", "value"))
def predict(input_text):
    if input_text is None:
        input_text = ""

    predictions = pd.DataFrame({
        "Confidence": np.array(model.predict_proba([[input_text]])).squeeze()[:, 1] * 100,
        "Genre": classes
    }).sort_values("Confidence", ascending=False).reset_index()

    return create_prediction_graph(predictions), create_interpretation_graph(input_text, predictions)


def create_prediction_graph(predictions):
    fig = px.bar(predictions[::-1], x="Confidence", y="Genre", range_x=[0, 100])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(fixedrange=True, ticksuffix="%", tickvals=np.linspace(0, 100, 11)),
        yaxis=dict(fixedrange=True, ticksuffix=" ", title=None)
    )
    fig.update_traces(hovertemplate="%{x:.1f}%")
    fig.update_xaxes(title_font_size=12)
    return dcc.Graph(figure=fig, config=dict(displayModeBar=False), responsive=True, style=dict(height="250px"))


def create_interpretation_graph(input_text, predictions):
    ordered_classes = list(predictions.Genre)
    tfidf_vector = tfidfvectorizer.transform([input_text]).toarray().squeeze()

    rows = 2
    cols = math.ceil(len(ordered_classes) / rows)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=predictions.apply(lambda x: f"{x.Genre} ({x.Confidence:.0f}%)", 1),
        horizontal_spacing=.015,
        vertical_spacing=.4
    )

    largest_contribution = 0
    for i in range(len(ordered_classes)):
        row, col = i // cols + 1, i % cols + 1

        contributions = coefficients.loc[ordered_classes[i]] * tfidf_vector
        contributions = contributions[contributions != 0]
        contributions = contributions[contributions.abs().nlargest(10).index]
        largest_contribution = max(largest_contribution, contributions.abs().max())

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
            fig.update_yaxes(title="Contribution", title_font_size=11, row=row, col=col)

    fig.update_annotations(font_size=13)
    fig.update_layout(
        hoverlabel_bordercolor="white",
        margin=dict(l=0, r=0, t=20, b=100),
        showlegend=False
    )
    fig.update_xaxes(
        fixedrange=True,
        showticklabels=bool(np.any(tfidf_vector)),
        tickfont=dict(size=11)
    )
    fig.update_yaxes(
        fixedrange=True,
        range=[-largest_contribution * 1.05, largest_contribution * 1.05],
        showticklabels=False,
        tickvals=[0]
    )

    return dcc.Graph(figure=fig, config=dict(displayModeBar=False), responsive=True, style=dict(height="400px"))


if __name__ == "__main__":
    app.run_server(debug=True)
