import pandas as pd
import numpy as np
import re
import warnings
import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

data_set = r'\\192.168.60.77\dataset_share\BAM_India_SPOT.xls'

df_=pd.read_excel(data_set)
df_.columns

#data cleaning

df_.drop(['Price series', 'Price Status', 'Unit of Measure', 'Low Price',
           'Mid Price'],axis =1 , inplace =True)

df_.rename(columns={'High Price':'dependent_variable'},inplace =True)

#arima (df_)

def arima (df_):
    import pandas as pd
    pattern = r'2023'

    df_ = df_[df_['Date'].dt.year.astype(str).str.contains(pattern, regex=True)]

    qty_mean = df_['dependent_variable'].mean()
    qty_std = df_['dependent_variable'].std()

    threshold = 3

    df = df_[abs(df_['dependent_variable'] - qty_mean) <= threshold * qty_std]

    df3=df
    df3['Date'] = pd.to_datetime(df3['Date'])

    df1=df3

    df1['Date'] = pd.to_datetime(df1['Date'])

    df2 = df1

    df2

    df2.plot()
    #testing for stationary
    

    #Ho: It is non stationary
    #H1: It is stationary
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']


    def adfuller_test(sales):
        result=adfuller(sales)
        for value,label in zip(result,labels):
            print(label+' : '+str(value) )
        if result[1] <= 0.05:
            print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
        else:
            print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")  


    adfuller_test(df2['dependent_variable'])
    result = adfuller(df2['dependent_variable'])  

    # p value is zera so we can say our data is stationary
    from pandas.plotting import autocorrelation_plot



    p_value = result[1]

    print(f"P-Value: {p_value}")
    ###### with first shift##########################################################################
    df2['Sales First Difference'] = df2['dependent_variable'] - df2['dependent_variable'].shift(1)
    df2['dependent_variable'].shift(1)

    df2['Seasonal First Difference']=df2['dependent_variable']-df2['dependent_variable'].shift(12)

    df2.head()

    ## Again test dickey fuller test
    adfuller_test(df2['Seasonal First Difference'].dropna())

    df2['Seasonal First Difference'].plot()
    ###########################################################################################


    ################################# with 2nd shiift#############################################
    df2['Seasonal 2nd difference'] = df2['Seasonal First Difference']-df2['Seasonal First Difference'].shift(1)
    df2['Seasonal 2nd difference'].shift(1)

    adfuller_test(df2['Seasonal 2nd difference'].dropna())


    ##################################################################################################
    #####                WITH 3RD SHIFT ##################################################################
    df2['Seasonal 3rd difference'] = df2['Seasonal 2nd difference']-df2['Seasonal 2nd difference'].shift(1)
    df2['Seasonal 3rd difference'].shift(1)

    adfuller_test(df2['Seasonal 3rd difference'].dropna())





    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df2['dependent_variable'])
    plt.show()

    from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df2['Seasonal First Difference'].iloc[13:],lags=10,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df2['Seasonal First Difference'].iloc[13:],lags=10,ax=ax2)
    '''

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df2['Seasonal 3rd difference'].iloc[3:],lags=6,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df2['Seasonal 3rd difference'].iloc[13:],lags=6,ax=ax2)


    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df2['Seasonal 2nd difference'].iloc[13:],lags=6,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df2['Seasonal 2nd difference'].iloc[13:],lags=6,ax=ax2)

    '''

    from statsmodels.tsa.arima.model import ARIMA


    #########################################################################################

    model = ARIMA(df2['dependent_variable'], order=(1, 2, 1))
    model_fit = model.fit()

    model_fit.summary()

    df2['forecast']=model_fit.predict(start=33,end=39,dynamic=True)
    df2[['dependent_variable','forecast']].plot(figsize=(12,8))
    ################################################################################################################


    '''
    import statsmodels.api as sm
    model=sm.tsa.statespace.SARIMAX(df2['dependent_variable'],order=(1,1,1),seasonal_order=(0,1,0,6))
    model_fit=model.fit()

    df2['forecast']=model_fit.predict(start=33,end=39,dynamic=True)
    df2[['dependent_variable','forecast']].plot(figsize=(12,8))
    '''
    from pandas.tseries.offsets import DateOffset
    import pandas as pd  # Ensure pandas is imported
    df2=df2.set_index('Date')
    # Convert the last date in the DataFrame index to a valid date object
    last_date = pd.to_datetime(df2.index[-1])

    # Calculate weekly date offsets
    future_dates = [last_date + DateOffset(weeks=x) for x in range(1, 20)]

    # Create an empty DataFrame for future dates
    future_datest_df = pd.DataFrame(index=future_dates, columns=df2.columns)

    future_datest_df=future_datest_df.reset_index()

    #future_datest_df.drop(columns={'Date'},inplace=True)

    future_datest_df=future_datest_df.rename(columns={'index':'Date'})

    df2=df2.reset_index()

    # Concatenate the original DataFrame with the future dates DataFrame
    #future_df = pd.concat([df2, future_datest_df])

    df2 = pd.concat([df2, future_datest_df])

    #future_df=future_df.reset_index()
    df2=df2.reset_index()
    df2.drop(columns='index',inplace=True)

    df2['forecast']=model_fit.predict(start=36,end=55,dynamic=True)


    
    
