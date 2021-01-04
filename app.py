import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px

model = joblib.load('model/model.joblib')
with open('model/metadata.json') as f:
    metadata = json.load(f)

app = dash.Dash(__name__)
server = app.server
app.title = 'Movie Genre Predictor'
app.layout = html.Div([
    html.H1('Movie Genre Predictor'),
    html.Div([
        dcc.Textarea(id='input', className='flex-item'),
        html.Figure(id='prediction', className='flex-item')
    ], className='flex-container')
])


@app.callback(Output('prediction', 'children'), Input('input', 'value'))
def predict(input):
    if input:
        pred = pd.DataFrame({
            'Confidence': np.array(model.predict_proba([[input]])).squeeze()[:,1],
            'Genre': metadata['classes']
        }).sort_values('Confidence')
        fig = px.bar(pred, x='Confidence', y='Genre', range_x=[0, 1])
        fig.update_layout(
            font_family='sans-serif',
            xaxis=dict(fixedrange=True, tickvals=np.linspace(0, 1, 11)),
            yaxis=dict(fixedrange=True, ticksuffix=' '))
        fig.update_traces(hovertemplate='%{x:.2f}')
        return dcc.Graph(figure=fig, config=dict(displayModeBar=False))


if __name__ == '__main__':
    app.run_server(debug=True)