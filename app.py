import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import joblib
import pandas as pd

model = joblib.load('model/model.joblib')
app = dash.Dash(__name__)
server = app.server
app.title = 'Movie Genre Predictor'
app.layout = html.Div([
    html.H1('Movie Genre Predictor'),
    html.Div([
        dcc.Textarea(id='input', className='flex-item'),
        html.Figure(className='flex-item')
    ], className='flex-container')
])

if __name__ == '__main__':
    app.run_server(debug=True)