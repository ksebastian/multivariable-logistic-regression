import dash
from dash import dcc, html
import plotly.graph_objs as go
import pickle
import json
from dash.dependencies import Input, Output, State
from ast import literal_eval

########### Define your variables ######
myheading1 = 'Iris Species Prediction'
image1 = '../assets/blueflagiris_flower.jpeg'
tabtitle = 'IRIS Flowers'
sourceurl = 'https://archive.ics.uci.edu/ml/datasets/iris'
githublink = 'https://github.com/ksebastian/multivariable-logistic-regression'

########### open the pickle file ######
filename = open('analysis/iris_multivariable_logistic_model.pkl', 'rb')
unpickled_model = pickle.load(filename)
filename.close()

########### list of feature values
### feature_names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server
app.title = tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),

    html.Div([
        html.Div(
            [html.Img(src=app.get_asset_url(image1), style={'width': '30%', 'height': 'auto'},
                      className='four columns')]),
        html.Div([
            html.H3("Features"),
            html.Div('Sepal Length(cm):'),
            dcc.Input(id='sepal_length', value=5.8, type='number', min=4.0, max=8.0, step=0.01),
            html.Div('Sepal Width(cm):'),
            dcc.Input(id='sepal_width', value=3.0, type='number', min=2.0, max=5.0, step=0.01),
            html.Div('Petal Length(cm):'),
            dcc.Input(id='petal_length', value=3.8, type='number', min=1.0, max=7.0, step=0.01),
            html.Div('Petal Width(cm):'),
            dcc.Input(id='petal_width', value=1.0, type='number', min=0.1, max=2.5, step=0.01),
        ], className='four columns'),
        html.Div([
            html.H3('Predictions'),
            html.Div('Probability of Setosa:'),
            html.Div(id='setosa_probability'),
            html.Br(),
            html.Div('Probability of Versicolor:'),
            html.Div(id='versicolor_probability'),
            html.Br(),
            html.Div('Probability of Virginica:'),
            html.Div(id='virginica_probability'),
            html.Br(),
        ], className='four columns')
    ], className='nine columns',
    ),

    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
]
)


######### Define Callback
@app.callback([Output(component_id='setosa_probability', component_property='children'),
               Output(component_id='versicolor_probability', component_property='children'),
               Output(component_id='virginica_probability', component_property='children'),
               ],
              [Input(component_id='sepal_length', component_property='value'),
               Input(component_id='sepal_width', component_property='value'),
               Input(component_id='petal_length', component_property='value'),
               Input(component_id='petal_width', component_property='value')
               ])
def prediction_function(sepal_length, sepal_width, petal_length, petal_width):
    try:
        data_list = [sepal_length, sepal_width, petal_length, petal_width]
        data = [data_list]
        print("Data:", data)

        setosa_prob = 100 * unpickled_model.predict_proba(data)[0][0]
        versicolor_prob = 100 * unpickled_model.predict_proba(data)[0][1]
        virginica_prob = 100 * unpickled_model.predict_proba(data)[0][2]
        print("Raw Probability:", setosa_prob, versicolor_prob, virginica_prob)

        return setosa_prob, versicolor_prob, virginica_prob
    except:
        return "inadequate inputs", "inadequate inputs", "inadequate inputs"


############ Deploy
if __name__ == '__main__':
    application.run(debug=True, port=8080)
