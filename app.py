import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, redirect, url_for, request
import json



data = pd.read_csv('train.csv', parse_dates=["date"])
test_data = pd.read_csv('test.csv', parse_dates=["date"])
data = data.set_index(data['date'])
data = data.sort_index()

#seperating data into traing and checking sets
train_data = data['2013-01-01':'2017-09-30']
check_data = data['2017-10-01':]

data.reset_index(drop=True, inplace=True)
train_data.reset_index(drop=True, inplace=True)
check_data.reset_index(drop=True, inplace=True)

###gets the total number of slaes for all stores and items for each day of the training data
sales_daily_train = train_data.groupby('date', as_index=False)['sales'].sum
sales_daily_check = check_data.groupby('date', as_index=False)['sales'].sum
sales_daily_all = data.groupby('date', as_index=False)['sales'].sum

### gets total sales by store for each day of training data
sales_daily_store_train = train_data.groupby(['store', 'date'], as_index=False)['sales'].sum()
sales_daily_store_check = check_data.groupby(['store', 'date'], as_index=False)['sales'].sum()
sales_daily_store_all = data.groupby(['store', 'date'], as_index=False)['sales'].sum()

### gets total sales of each item for each day of traing set
sales_daily_item_train = train_data.groupby(['item', 'date'], as_index=False)['sales'].sum()
sales_daily_item_check = check_data.groupby(['item', 'date'], as_index=False)['sales'].sum()
sales_daily_item_all = data.groupby(['item', 'date'], as_index=False)['sales'].sum()

### gets total sales for all items and stores by day
sales_daily_total_train = train_data.groupby(['date'], as_index=False)['sales'].sum()
sales_daily_total_check = check_data.groupby(['date'], as_index=False)['sales'].sum()
sales_daily_total_all = data.groupby(['date'], as_index=False)['sales'].sum()

# create a copy of our data to enrich
td_temp = train_data.copy()
test_temp = test_data.copy()
check_temp = check_data.copy()
data_temp = data.copy()

# clean our data for processing and seperate out our dates into individual columns
td_temp["saleYear"] = train_data.date.dt.year
td_temp["saleMonth"] = train_data.date.dt.month
td_temp["saleDay"] = train_data.date.dt.day
td_temp["saleDayOfWeek"] = train_data.date.dt.dayofweek
if "date" in td_temp.columns:
    td_temp.drop("date", axis=1, inplace=True)

test_temp["saleYear"] = test_data.date.dt.year
test_temp["saleMonth"] = test_data.date.dt.month
test_temp["saleDay"] = test_data.date.dt.day
test_temp["saleDayOfWeek"] = test_data.date.dt.dayofweek
if "date" in test_temp.columns:
    test_temp.drop("date", axis=1, inplace=True)
if "id" in test_temp.columns:
    test_temp.drop("id", axis=1, inplace=True)

check_temp["saleYear"] = check_data.date.dt.year
check_temp["saleMonth"] = check_data.date.dt.month
check_temp["saleDay"] = check_data.date.dt.day
check_temp["saleDayOfWeek"] = check_data.date.dt.dayofweek
if "date" in check_temp.columns:
    check_temp.drop("date", axis=1, inplace=True)

df_preds = pd.read_csv('predsnormal.csv', parse_dates=["date"])
df_check_preds = pd.read_csv('checkpreds.csv', parse_dates=["date"])
df_long_preds = pd.read_csv('predslong.csv', parse_dates=["date"])


def singlePred(item, store, year, day, month):
    years = str(year)
    days = str(day)
    months = str(month)
    date = years + '-' + months + '-' + days
    returndf = pd.DataFrame
    returndf = df_long_preds.loc[(df_long_preds['date'] == date) & (df_long_preds['item'] == item) & (df_long_preds['store'] == store)]
    return str(int(returndf['sales'])) + " items"


###gets the total number of slaes for all stores and items for each day of the training data
preds_sales_daily = df_preds.groupby('date', as_index=False)['sales'].sum()
check_preds_sales_daily = df_check_preds.groupby('date', as_index=False)['sales'].sum()
test_sales_daily = check_data.groupby('date', as_index=False)['sales'].sum()

### gets total sales by store for each day of training data
preds_sales_daily_store = df_preds.groupby(['store', 'date'], as_index=False)['sales'].sum()

### gets total sales of each item for each day of traing set
preds_sales_daily_item = df_preds.groupby(['item', 'date'], as_index=False)['sales'].sum()

#check line plots
psd = preds_sales_daily.copy()
psd = psd.assign(Data_Types='Future Predictions')

cpsd = check_preds_sales_daily.copy()
cpsd = cpsd.assign(Data_Types='Test Prediction')

tsd = test_sales_daily.copy()
tsd = tsd.assign(Data_Types='Test Data')

frames = [psd, cpsd, tsd]

fig = px.line(pd.concat(frames), x="date", y="sales", color = "Data_Types" )

def plotStoreTest(stores):
    storePlot = []
    for store in stores['store'].unique():
        current_store_daily_sales = stores[(stores['store'] == store)]
        storePlot.append(go.Scatter(x=current_store_daily_sales['date'], y=current_store_daily_sales['sales'], name=('Store %s' % store)))
    layout = go.Layout(title='Daily Store Sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
    fig = go.Figure(data=storePlot, layout=layout)
    return fig

def plotStorePred(stores):
    storePlot = []
    for store in stores['store'].unique():
        current_store_daily_sales_preds = stores[(stores['store'] == store)]
        storePlot.append(go.Scatter(x=current_store_daily_sales_preds['date'], y=current_store_daily_sales_preds['sales'], name=('Store %s' % store)))
    layout = go.Layout(title='Daily Store Sales Prediction', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
    fig = go.Figure(data=storePlot, layout=layout)
    return fig

def plotItems(items):
    itemsPlot = []
    for item in items['item'].unique():
        current_item_daily_sales = items[(items['item'] == item)]
        itemsPlot.append(go.Scatter(x=current_item_daily_sales['date'], y=current_item_daily_sales['sales'], name=('Item %s' % item)))
    layout = go.Layout(title='Daily Item Sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
    fig = go.Figure(data=itemsPlot, layout=layout)
    return fig

def plotItemPreds(items):
    preds = []
    for item in items['item'].unique():
        current_item_daily_sales = items[(items['item'] == item)]
        preds.append(go.Scatter(x=current_item_daily_sales['date'], y=current_item_daily_sales['sales'], name=('Item %s' % item)))
    layout = go.Layout(title='Predicted Daily Item Sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
    fig = go.Figure(data=preds, layout=layout)
    return fig

app = Flask(__name__)

@app.route("/home")
def home():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'wguadmin' or request.form['password'] != 'wguadmin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    error = None
    if request.method == 'POST':
        if int(request.form['year']) < 2017 or int(request.form['year']) > 2022:
            error = 'Invalid data must be after 2017 and before 2022'
        elif int(request.form['day']) < 1 or int(request.form['day']) >31:
            error = 'Invalid Day was entered'
        elif int(request.form['month']) < 1 or int(request.form['month']) > 12:
            error = 'Invalid Month was entered'
        elif int(request.form['item']) < 1 or int(request.form['item']) > 50:
            error = 'Invalid item was entered'
        elif int(request.form['store']) < 1 or int(request.form['store']) > 10:
            error = 'Invalid store was entered'
        else:
            error = singlePred(int(request.form['item']), int(request.form['store']), int(request.form['year']), int(request.form['day']), int(request.form['month']))
    return render_template('prod.html', error=error)

@app.route("/storeSales")
def storeSales():
    fig = plotStoreTest(sales_daily_store_all)
    graphJSON = json.dumps(fig, cls=py.utils.PlotlyJSONEncoder)
    return render_template('test.html', graphJSON=graphJSON)

@app.route("/storeSalesPreds")
def storeSalesPreds():
    fig = plotStorePred(preds_sales_daily_store)
    graphJSON = json.dumps(fig, cls=py.utils.PlotlyJSONEncoder)
    return render_template('test.html', graphJSON=graphJSON)

@app.route("/itemSales")
def itemSales():
    fig = plotItems(sales_daily_item_all)
    graphJSON = json.dumps(fig, cls=py.utils.PlotlyJSONEncoder)
    return render_template('test.html', graphJSON=graphJSON)

@app.route("/itemSalesPreds")
def itemSalesPreds():
    fig = plotItemPreds(preds_sales_daily_item)
    graphJSON = json.dumps(fig, cls=py.utils.PlotlyJSONEncoder)
    return render_template('test.html', graphJSON=graphJSON)

@app.route("/mlModel")
def mlModel():
    fig = px.line(pd.concat(frames), x="date", y="sales", color = "Data_Types" )
    graphJSON = json.dumps(fig, cls=py.utils.PlotlyJSONEncoder)
    return render_template('test.html', graphJSON=graphJSON)

if __name__=='__main__':
    app.run()
