import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from dateutil.parser import parse
from pylab import rcParams
import seaborn as sns
import os

# Setting up Streamlit configurations
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")

# Set the page configuration at the top of the script
st.set_page_config(page_title="Time Series Analysis",
                   page_icon="🚀",
                   layout="wide",
                   initial_sidebar_state="expanded")

def find_outlier(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier = data[(data < lower) | (data > upper)]
    return outlier

def adf_test(timeseries):
    st.write('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    st.write(dfoutput)

# Define each page as a function
def ARIMA_page():
    st.title("ARIMA")
    st.write("Let's using the ARIMA")
        
    namelist = st.multiselect('Select countries to analyze:', ['united', 'china', 'japan'])

    for i in namelist:
        st.markdown(f'## {i.capitalize()} Analysis')
        
        data = pd.read_csv(f'files/{i}.csv')
        data_cleaned = data.drop(['Unnamed: 0', 'Country'], axis=1)
        
        st.write(data_cleaned.head())
        
        # Plotting the data
        st.write("### Data Plot")
        st.line_chart(data_cleaned)
        
        # Data transformations
        data_cleaned['Month'] = data_cleaned['Month'].astype(str)
        data_cleaned['Year'] = data_cleaned['Year'].astype(str)
        data_cleaned['Month Year'] = data_cleaned['Month'].str.zfill(2) + '-' + data_cleaned['Year']
        data_datetime = data_cleaned.set_index(pd.to_datetime(data_cleaned['Month Year'], format='%m-%Y', errors='coerce'))
        
        # Seasonal decomposition
        st.write("### Seasonal Decomposition")
        decomposition = sm.tsa.seasonal_decompose(data_datetime['Value (million baht)'], model='additive', period=12)
        
        # Plotting the decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 8))
        decomposition.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')

        # Scatter plot for residuals
        ax4.scatter(data_datetime.index, decomposition.resid, color='darkblue', marker='o')
        ax4.axhline(0, color='darkblue', linestyle='--')
        ax4.set_ylabel('Residual')
        
        st.pyplot(fig)

        # Introduction to ARIMA model analysis
        st.markdown("## Analysis for the ARIMA model using pmdarima")

        # Fit auto_arima function to dataset
        stepwise_fit = auto_arima(data_datetime['Value (million baht)'], start_p=1, start_q=1,
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=None, D=1, trace=True,
                                error_action='ignore',  # we don't want to know if an order does not work
                                suppress_warnings=True,  # we don't want convergence warnings
                                stepwise=True)  # set to stepwise

        # Display the ARIMA model parameters
        st.markdown("### Selected ARIMA Model Parameters")
        st.write(stepwise_fit.summary())

        # Splitting the data into train and test sets
        train_data = data_datetime.iloc[:len(data_datetime)-12]
        test_data = data_datetime.iloc[len(data_datetime)-12:]

        # Fit a SARIMAX model
        st.markdown("### Fitting the SARIMAX Model")
        model = SARIMAX(train_data['Value (million baht)'],
                        order=(1, 0, 1),
                        seasonal_order=(2, 1, 2, 12))
        result = model.fit()

        # Display the SARIMAX model summary
        st.write(result.summary())
        # Actual data
        actual = data_datetime['Value (million baht)']

        # Fitted values from the model
        fitted = result.fittedvalues

        # Plotting Actual vs Fitted values
        st.markdown("### Actual vs Fitted Values")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual, label='Actual')
        ax.plot(fitted, label='Fitted', color='red')
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Value (million baht)')
        ax.set_title('Actual vs Fitted')
        st.pyplot(fig)

        # Forecasting for the next 1 year
        st.markdown("### Forecast for the Next 1 Year")
        forecast = result.predict(start='2022-07-01', end='2023-06-01', typ='levels').rename('Forecast')

        # Plotting the forecasted values and the actual values
        fig, ax = plt.subplots(figsize=(12, 5))
        data_datetime['Value (million baht)'].plot(ax=ax, label='Actual', legend=True)
        forecast.plot(ax=ax, label='Forecast', legend=True)
        ax.set_xlabel('Year')
        ax.set_ylabel('Value (million baht)')
        ax.set_title(f'Forecasted vs. Actual Values for {i.capitalize()}')
        st.pyplot(fig)

        # Calculating error metrics
        test_values = test_data['Value (million baht)']
        mape = (np.abs(test_values - forecast) / test_values).mean() * 100
        mse = ((forecast - test_values) ** 2).mean()
        rmse = np.sqrt(mse)

        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Forecasting for the next 5 years
        st.markdown("### Forecast for the Next 5 Years")
        forecast_total = result.predict(start=len(data_datetime), end=(len(data_datetime) - 1) + 5 * 12, typ='levels').rename('Forecast')

        # Plotting the forecasted values and the actual values
        fig, ax = plt.subplots(figsize=(12, 5))
        data_datetime['Value (million baht)'].plot(ax=ax, label='Actual', legend=True)
        forecast_total.plot(ax=ax, label='Forecast', legend=True)
        ax.set_xlabel('Year')
        ax.set_ylabel('Value (million baht)')
        ax.set_title(f'Forecasted vs. Actual Values for {i.capitalize()} (Next 5 Years)')
        st.pyplot(fig)

def SARIMA_page():
    # Title and introduction
    st.title("SARIMA Forecasting")
    st.write("Let's use the SARIMA model to forecast export values.")

    namelist = st.multiselect('Select countries to analyze:', ['united states', 'china', 'japan'])

    for i in namelist:
        st.markdown(f'## {i.capitalize()} Analysis')
        
        data = pd.read_csv(f'files/{i}.csv')
        data_cleaned = data.drop(['Unnamed: 0', 'Country'], axis=1)
        
        st.write(data_cleaned.head())
        
        # Plotting the data
        st.write("### Data Plot")
        st.line_chart(data_cleaned)
        
        # Transform data for SARIMA
        data_cleaned['Month'] = data_cleaned['Month'].astype(str)
        data_cleaned['Year'] = data_cleaned['Year'].astype(str)
        data_cleaned['Month Year'] = data_cleaned['Month'].str.zfill(2) + '-' + data_cleaned['Year']
        data_datetime = data_cleaned.set_index(pd.to_datetime(data_cleaned['Month Year'], format='%m-%Y', errors='coerce'))
        
        # Copying the data
        data_cleaned_new = data_datetime.copy()
        
        # First difference
        data_cleaned_new['Value (million baht)_diff'] = data_cleaned_new['Value (million baht)'].diff(1)

        # Time Series Decomposition
        rcParams['figure.figsize'] = (18, 8)
        decomposition = sm.tsa.seasonal_decompose(data_cleaned['Value (million baht)'], model='additive', period=12)
        fig = decomposition.plot()
        st.pyplot(fig)

        # Plotting the first difference
        st.write("### First Difference Plot")   
        color_diff = st.color_picker('Chart color:')
        st.line_chart(data_cleaned_new['Value (million baht)_diff'],color=color_diff)

        # Selecting specific columns
        columns = ['Value (million baht)_diff', 'Value (million baht)']
        data_selected_columns = data_cleaned_new[columns]

        # Ensure DateTime Index is Sorted
        data_selected_columns = data_selected_columns.sort_index()
        
        st.write("### Selected Data")
        st.write(data_selected_columns.head())

        # Dividing data into training and testing sets
        train_data = data_selected_columns.iloc[:125]
        test_data = data_selected_columns.iloc[125:]
        # Dropping NaN values before fitting the model
        train_data = train_data.dropna()

        # Parameter grid for SARIMA
        p = d = q = range(0, 2)
        P = D = Q = range(0, 2)
        param_grid = dict(p=p, d=d, q=q, P=P, D=D, Q=Q)

        # Finding the best SARIMA model
        best_aic = float('inf')
        best_model = None
        best_param = None
        for p in param_grid['p']:
            for d in param_grid['d']:
                for q in param_grid['q']:
                    for P in param_grid['P']:
                        for D in param_grid['D']:
                            for Q in param_grid['Q']:
                                try:
                                    model = SARIMAX(train_data['Value (million baht)_diff'].dropna(),
                                                    order=(p, d, q),
                                                    seasonal_order=(P, D, Q, 12),
                                                    enforce_stationarity=False)
                                    results = model.fit()
                                    if results.aic < best_aic:
                                        best_aic = results.aic
                                        best_model = results
                                        best_param = (p, d, q, P, D, Q)
                                except:
                                    continue

        # Displaying the best parameters and AIC
        st.write("### Best Parameters")
        st.write(f"Best Parameters: {best_param}")
        st.write(f"Best AIC: {best_aic}")   

        # Predicting the values for the test data
        if best_model is not None and not test_data.empty:  # Added check for test_data not being empty
            try:
                y_pred = best_model.get_prediction(start=test_data.index[0], end=test_data.index[-1], dynamic=False)
                predictions = y_pred.predicted_mean

                # Calculating the error metrics
                mae = mean_absolute_error(test_data['Value (million baht)_diff'].dropna(), predictions.dropna())  # Added dropna to handle NaN values
                mse = mean_squared_error(test_data['Value (million baht)_diff'].dropna(), predictions.dropna())  # Added dropna to handle NaN values
                rmse = mse ** 0.5

                # Displaying the error metrics
                st.write("### Error Metrics")
                st.write(f"Mean Absolute Error (MAE): {mae}")
                st.write(f"Mean Squared Error (MSE): {mse}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse}")

                # Creating a SARIMAX model with specific parameters
                mod = sm.tsa.statespace.SARIMAX(data_selected_columns['Value (million baht)'],
                                                # order=best_param[:3],  # Using the best parameters found in the earlier step
                                                # seasonal_order=best_param[3:] + (12,),  # Adding 12 for the seasonal period
                                                # enforce_invertibility=False)
                                                order=(0,1,1),  # Using the best parameters found in the earlier step
                                                seasonal_order=(0,1,1,12),  # Adding 12 for the seasonal period
                                                enforce_invertibility=False)
                results = mod.fit()

                # Plotting diagnostics
                st.write("### Model Diagnostics")
                fig = plt.figure(figsize=(16, 10))
                best_model.plot_diagnostics(fig=fig, lags=30)  # Adjust the lags as necessary
                plt.tight_layout()  # Adjust layout to prevent overlap
                st.pyplot(fig)

                # One-step ahead Predictions
                st.write("### One-Step Ahead Predictions")
                pred = results.get_prediction(start=pd.to_datetime('2023-01-01'), dynamic=False)
                pred_ci = pred.conf_int()

                fig, ax = plt.subplots(figsize=(14, 7))

                # Plotting the observed data
                data_selected_columns['Value (million baht)']['2013':].plot(ax=ax, label='Observed')

                # Plotting the one-step ahead predictions
                pred.predicted_mean.plot(ax=ax, label='One-step Ahead Predictions', alpha=.7)

                # Filling the confidence interval
                ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0],
                                pred_ci.iloc[:, 1], color='k', alpha=.2)

                ax.set_xlabel('Date')
                ax.set_ylabel('Value (million baht)')
                ax.set_title(f'{i.capitalize()} One-Step Ahead Predictions')
                plt.legend()

                st.pyplot(fig)

                # One-step ahead Predictions from 2013 onwards
                st.write("### One-Step Ahead Predictions from 2013 Onwards")

                # Check if the start date exists in the DataFrame index
                start_date = pd.to_datetime('2013-01-01')
                if start_date in data_selected_columns.index:
                    st.write(f"The date {start_date} is in the index.")
                else:
                    st.write(f"The date {start_date} is NOT in the index.")
                if start_date in data_selected_columns.index:
                    try:
                        # Getting predictions
                        pred_from_2013 = results.get_prediction(start=start_date, dynamic=False)
                        pred_ci_from_2013 = pred_from_2013.conf_int()

                        # Creating a plot
                        fig, ax = plt.subplots(figsize=(14, 7))

                        # Plotting the observed data
                        data_selected_columns['Value (million baht)']['2013':].plot(ax=ax, label='Observed')

                        # Plotting the one-step ahead predictions from 2013 onwards
                        pred_from_2013.predicted_mean.plot(ax=ax, label='One-step Ahead Predictions from 2013', alpha=.7)

                        # Diagnostic: Print out the first few predictions to the console
                        st.write("First few predictions:")
                        st.write(pred_from_2013.predicted_mean.head())

                        # Filling the confidence interval
                        ax.fill_between(pred_ci_from_2013.index,
                                        pred_ci_from_2013.iloc[:, 0],
                                        pred_ci_from_2013.iloc[:, 1], color='k', alpha=.2)

                        ax.set_xlabel('Date')
                        ax.set_ylabel('Value (million baht)')
                        ax.set_title(f'{i.capitalize()} One-Step Ahead Predictions from 2013 Onwards')
                        plt.legend()

                        # Display the plot in Streamlit
                        st.pyplot(fig)
                    except Exception as e:
                        # Handle any exceptions that occur during prediction and plotting
                        st.write(f"An error occurred: {e}")

                    # Print the model summary
                    st.write(results.summary())

                    # Print the start date and end date
                    st.write(f"Start Date: {start_date}")
                    st.write(f"End Date: {data_selected_columns.index[-1]}")

                    # Check for NaN or Inf values in the data
                    st.write(f"NaN values in the data: {data_selected_columns.isna().sum()}")
                    st.write(f"Inf values in the data: {np.isinf(data_selected_columns).sum()}")

                else:
                    st.write(f"The start date {start_date.strftime('%Y-%m-%d')} is not in the data index.")

                # Displaying the model summary
                st.write("### Model Summary")
                st.text(str(results.summary().tables[1]))  # Using st.text to display the summary table as plain text

            except IndexError:
                st.write("Unable to make predictions due to index error.")

        elif test_data.empty:
            st.write("The test data is empty, unable to make predictions.")


def SUM_page():
        st.title("SUM")
        st.write("Let's using the SUMMARY")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import GridSearchCV

def DEMO_page():
        st.title("Demo")

        df_demo = pd.read_csv('iris.data')
        st.dataframe(df_demo)

        X = df_demo.iloc[:,:-1]
        y = df_demo.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.3, 
                                                            random_state=0)

        classifer = RandomForestClassifier(n_estimators=5, random_state=0)
        
        # n_list = np.arange(1,100,1, dtype='int')
        # parameters = {
        #     'n_estimators' : n_list
        # }

        # grid = GridSearchCV(estimator= classifer,
        #                     param_grid = parameters,
        #                     cv=5,
        #                     scoring='accuracy')
        # grid.fit(X_train, y_train)
        # y_pred = grid.predict(X_test)

        classifer.fit(X_train, y_train)
        y_pred = classifer.predict(X_test)

        new_y_test_1sample = [[10,0.2,0.5,1.0]]
        y_pred2 = classifer.predict(new_y_test_1sample)

        score = accuracy_score(y_test, y_pred)

        pickle_out = open('model_iris.pkl','wb')
        pickle.dump(classifer, pickle_out)
        pickle_out.close()

        col1, col2 = st.columns(2)
        # col1.markdown(f''' # n_estimators: ''')
        # col1.write(n_list)
        col1.write(y_pred2)
        
        col2.markdown(f''' # score: {round(score,3)*100}%''')
        # col2.markdown(f''' # score: {round(score,3)*100}%
        #              best parameter: {grid.best_params_}''')

from PIL import Image

def Iris_app():
    st.title("IRIS")
    st.write('Iris Application')

    warnings.filterwarnings("ignore")

    pickle_in = open("model_iris.pkl", "rb")
    classifier = pickle.load(pickle_in)

    def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
        prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        print(prediction)
        return prediction

    def input_output():
        st.title("Iris Variety Prediction")
        st.image("https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png", width=600)
        st.markdown("You are using Streamlit...", unsafe_allow_html=True)

        sepal_length = (st.text_input("Enter Sepal Length", "."))
        sepal_width = (st.text_input("Enter Sepal Width", "."))
        petal_length = (st.text_input("Enter Petal Length", "."))
        petal_width = (st.text_input("Enter Petal Width", "."))       
    
        result = ""
        if st.button("Click here to Predict"):
            result = predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width)
            st.snow()
            st.success(f'The output is {result}')

    if __name__ == '__main__':
        input_output()

    


# Create a sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["ARIMA", "SARIMA","SUM","Demo","IRIS"])

# Display the selected page
if page == "ARIMA":
    ARIMA_page()
elif page == "SARIMA":
    SARIMA_page()
elif page == "SUM":
    SUM_page()
elif page == "Demo":
    DEMO_page()
elif page == "IRIS":
    Iris_app()


