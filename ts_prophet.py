import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def prophet_forecasting(file_path, output_file_path, start_year_column, num_rows=None):
    # Load the uploaded Excel file
    data = pd.read_excel(file_path)

    # Fill NaN values using forward fill method
    data_filled = data.ffill()

    # If num_rows is specified, select the first num_rows rows
    if num_rows:
        data_filled = data_filled.head(num_rows)

    # Re-initialize lists to store predictions and MAPE values
    pred_2020, pred_2021, pred_2022 = [], [], []
    mape_2020, mape_2021, mape_2022 = [], [], []

    # Iterate over each row in the dataframe
    for index, row in data_filled.iterrows():
        print(f"Processing row {index + 1}/{len(data_filled)}")
        
        # Extract the time series data starting from the specified column to 2019
        ts = row.iloc[start_year_column:row.index.get_loc('2019')+1].astype(float)
        ts.index = pd.to_datetime(ts.index, format='%Y')

        # Prepare the data for Prophet
        df = pd.DataFrame({'ds': ts.index, 'y': ts.values})

        # Fit Prophet model
        model = Prophet()
        model.fit(df)

        # Make future dataframe for 2020, 2021, and 2022
        future = model.make_future_dataframe(periods=4, freq='Y')
        forecast = model.predict(future)

        # Check if the forecast contains the required dates
        print(forecast[['ds', 'yhat']])  # Print forecast to debug

        try:
            forecast_2020 = forecast[forecast['ds'] == '2020-12-31']['yhat'].values[0]
        except IndexError:
            forecast_2020 = None
        
        try:
            forecast_2021 = forecast[forecast['ds'] == '2021-12-31']['yhat'].values[0]
        except IndexError:
            forecast_2021 = None
        
        try:
            forecast_2022 = forecast[forecast['ds'] == '2022-12-31']['yhat'].values[0]
        except IndexError:
            forecast_2022 = None

        pred_2020.append(forecast_2020)
        pred_2021.append(forecast_2021)
        pred_2022.append(forecast_2022)
        
        # Calculate MAPE for each year if forecast is available
        if forecast_2020 is not None:
            mape_2020.append(mean_absolute_percentage_error([row['2020']], [forecast_2020]))
        else:
            mape_2020.append(None)
        
        if forecast_2021 is not None:
            mape_2021.append(mean_absolute_percentage_error([row['2021']], [forecast_2021]))
        else:
            mape_2021.append(None)
        
        if forecast_2022 is not None:
            mape_2022.append(mean_absolute_percentage_error([row['2022']], [forecast_2022]))
        else:
            mape_2022.append(None)

    # Add predictions and MAPE values to the dataframe
    data_filled['2020_prophet_pred'] = pred_2020
    data_filled['2021_prophet_pred'] = pred_2021
    data_filled['2022_prophet_pred'] = pred_2022
    data_filled['2020_prophet_MAPE'] = mape_2020
    data_filled['2021_prophet_MAPE'] = mape_2021
    data_filled['2022_prophet_MAPE'] = mape_2022

    # Save the updated dataframe to a new Excel file
    data_filled.to_excel(output_file_path, index=False)

# Example usage
file_path = 'sample_data_set.xlsx'
output_file_path = 'updated_sample_data_set_prophet_test.xlsx'
start_year_column = int(input("Please enter the column number where the time series starts: "))
prophet_forecasting(file_path, output_file_path, start_year_column, num_rows=10)
