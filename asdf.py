# For processing data
import pandas as pd
import numpy as np
import math
from dateutil.parser import parse
import time

# API keys
import keys

# For API access
from bittrex.bittrex import Bittrex


def get_SMA(close_data, time_period):
    """Computes simple moving average (SMA) for the specified time_period.
    
    Parameters
    ----------
    
    close_data: Pandas Series
        Pandas Series object containing the close data (1-dimensional)
        
    time_period : int
        Number of days to consider for the SMA
    
    Returns
    ----------
    SMA: Pandas Series
        Pandas Series object that contains the simple moving average for the
        close_data.
        
    
    Example
    -------
   To compute a 10-day SMA for the close_data:
           
    # Compute 10-day SMA
    SMA10 = get_SMA(close_data, 10)
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Date: August, 31, 2019
    """
        
    # List to store moving average results
    SMA = list(range(0, len(close_data) - time_period))
    
    # Compute moving average
    for ii in range(len(SMA)):
        # Previous days index
        index = range(ii, ii + time_period)
        
        # Get data for previous days
        prev_days = close_data.iloc[index]
        
        # Sum previous days
        summation = np.sum(prev_days)
        
        # Get average
        avg = summation/time_period
        
        # Save results to list
        SMA[ii] = avg
         
    # Define column label
    label = f"{time_period}-SMA"
    
    # Get corresponding dates for moving_avg
    dates = close_data.index[time_period:]
    
    # Convert list into Pandas Series
    SMA = pd.Series(SMA, name = label, index = dates)
    
    return SMA


def get_EMA(close_data, time_period):
    """Computes exponential moving average (EMA) for the specified time_period.
    
    Parameters
    ----------
    
    close_data: Pandas Series
        Pandas Series object containing the close data (1-dimensional)
        
    time_period : int
        Number of days to consider for the SMA
    
    Returns
    ----------
    EMA: Pandas Series
        Pandas Series object that contains the exponential moving average for the
        close_data.
        
    
    Example
    -------
   To compute a 10-day EMA for the close_data:
           
    # Compute 10-day EMA
    EMA10 = get_EMA(close_data, 10)
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Date: August, 31, 2019
    """
    
    # List to store moving average results
    EMA = list(range(0, len(close_data) - time_period))
    
    # Calculate SMA to use as the first EMA
    initial_EMA = get_SMA(close_data, time_period)[0]
    
    # Calculate initial weight
    k = 2.0 / (time_period + 1)

    # Compute EMA
    for ii in range(len(EMA)):
        # Set index
        index = time_period + ii
        
        # Get current Close price
        close_temp = close_data[index]
        
        # Compute current EMA
        if ii == 0:
            EMA_temp = (close_temp - initial_EMA)*k + initial_EMA
        else:
            EMA_temp = (close_temp - EMA[ii-1])*k + EMA[ii-1]


        # Save results to list
        EMA[ii] = EMA_temp
        
    # Define column label
    label = f"{time_period}-EMA"
    
    # Get corresponding dates for moving_avg
    dates = close_data.index[time_period:]
    
    # Convert list into Pandas Series
    EMA = pd.Series(EMA, name = label, index = dates)
    
    return EMA


def getDailyVol(close, span0):
    """Computes the daily votality at intraday estimation points, applying a 
    span of span0 days to an exponentially weighted moving standard deviation.
    The original source code was written by Marcos and appears in its original 
    form in his book titled Advances in Financial Machine Learning. It was
    adapted edited by Frank. In the original version there is a bug.
    
    Parameters
    ----------
    
    close: Pandas Series,
        Pandas series containing close prices.
    
    span0: int,
        Integer number determiningthe span of the exponential weighted average.
        
    
    Returns
    ----------
    ewmsd: Pandas Series
        Expontentially weighted moving standard deviation of daily price
        returns.
        
    
    Example
    -------
   To compute daily volality:
           
       daily_vol = getDailyVol(close = close, span0 = 100)
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Marcos Lopez de Prado
    LinkedIn: <https://www.linkedin.com/in/lopezdeprado/>
    
    Date: September 13, 2019
    """
    
    # Initiate price return list
    price_returns = []
    dates = []
    
    # Calculate daily price returns
    for today_date, next_date, next_next_date in zip(close.index[:-2], close.index[1:-1], close.index[2:]):
        # Daily price return 
        temp_return = (close.loc[next_next_date]/close.loc[next_date]) - 1
        
        # Save results
        price_returns.append(temp_return)
        dates.append(today_date)
    
    # List to Pandas Series
    price_returns = pd.Series(price_returns, index = dates)
    
    # Calculate exponentially weighted moving standard deviation
    ewmsd = price_returns.ewm(span = span0).std()

    return ewmsd


def getFixedTimeHorizonLabels(close, horizon, span0):
    """Labels observations using the fixed-time horizon method. Each observation
    is labeled based using a rolling exponetially weighted standard deviation (ewmsd) of
    the daily price returns. For example, suppose that the ewmsd was 0.5, then we
    will label our observations as follows:
    
        y =  -1 if price return < - 0.5
              0 if abs(price return) <= 0.5
              1 if price return > 0.5
    
    The price return is calculated over a bar horizon as:
    
        priceReturn = price(at end of horizon)/price(at beginning of horizon) - 1
    
    Parameters
    ----------
    
    close: Pandas Series,
        Pandas series containing close prices.
    
    horizon: int,
        Integer number determining the number of bars in the fixed-time horizon. 
        
     span0: int,
        Integer number determining the span of the exponential weighted average.
        
    Returns
    ----------
    labels: Pandas Series
        Labels based on fixed time horizon.
        
    
    Example
    -------
           
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Marcos Lopez de Prado
    LinkedIn: <https://www.linkedin.com/in/lopezdeprado/>
    
    Date: November 1, 2019
    """
    
    # Initiate price return list
    dailyVol = getDailyVol(close, span0)
    labels = []
    dates = []
    price_returns = []
    
    # Calculate daily price returns
    for before_Hdate, first_Hdate, last_Hdate in zip(close.index[:-horizon], close.index[1:-(horizon)+1], close.index[horizon:]):
        # Daily price return
        temp_return = (close.loc[last_Hdate]/close.loc[first_Hdate]) - 1
        
        if not math.isnan(dailyVol[before_Hdate]) :
            # Determine label
            if temp_return  < -dailyVol[before_Hdate]:
                temp_label = -1
            elif np.abs(temp_return) <= dailyVol[before_Hdate]:
                temp_label = 0
            elif temp_return > dailyVol[before_Hdate]:
                temp_label = 1
                
            # Append results
            labels.append(temp_label)
            dates.append(before_Hdate)
            price_returns.append(temp_return)
    
    # List to Pandas Series
    labels = pd.Series(labels, index = dates)
    price_returns = pd.Series(price_returns, index =dates)
    
    return labels


def getUSDMarketLabels():
    """Returns a list of all USD markets available in Bittrex. This only works
    with the 1.1 version of the Bittrex API.

    
    Parameters
    ----------
        None
        
    Returns
    ----------
    market_labels: list
        List containing all the USD markets available at Bittrex.
        
    
    Example
    -------
        None    
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Date: November 20, 2019
    """
    
    # Create bittrex object
    my_bittrex = Bittrex(api_key = keys.api_key, api_secret = keys.api_secret,
                     api_version='v2.0') # v1.1
    
    # Get market summaries
    summaries = my_bittrex.get_market_summaries()['result']
  
    # Get market labels
    market_labels = []
    
    for summary in summaries:
        if 'USD-' in summary['MarketName']:
            market_labels.append(summary['MarketName'])
            
    return market_labels



def getUSDMarketData(time_period):
    """For every USD market returned by getUSDMarketLabels(), grab the last 100
    transactions about every two minutes. Then for each market, create a .csv file.
    This function is meant to create a data base. The data will be continously 
    collected for duration of the time_period in hours.

    
    Parameters
    ----------
    time_period, integer 
        The period of time to collect in hours. 
        
    Returns
    ----------
    None
        
    
    Example
    -------
        time_period = 18 % hours
        getUSDMarketData(time_period)   
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Date: December 10, 2019
    """
    
    # Get USD market labels
    market_labels = getUSDMarketLabels()
    
    # Create dictionary to store data
    market_data = {market_label: [] for market_label in market_labels}
    
    # Get time now
    start_time = time.time()
    
    # Define total time to get data
    total_time = time_period*60*60 # seconds
    
    # Create bittrex object
    my_bittrex = Bittrex(api_key = keys.api_key, api_secret = keys.api_secret,
                     api_version='v1.1')
    
    while time.time() - start_time < total_time:
        print(f'The amount of time elapsed is: {time.time() - start_time}')
        for market_label in market_labels:
            # Message to user
            print(f'Currently grabbing data for {market_label}...')
            
            # Get market history
            market_history = my_bittrex.get_market_history(market_label)['result']

            # Keys to remove from each transaction
            remove_keys = ['FillType', 'OrderType', 'Total']
            
            # Remove uncessary data
            clean_market_history = []
            for transaction in market_history:
                for remove_key in remove_keys:
                    transaction.pop(remove_key, None)
                clean_market_history.append(transaction)
                
            # Append data
            market_data[market_label] = market_data[market_label] + clean_market_history 
            
            # Wait for 5 seconds
            time.sleep(5)
    
    # Save the results        
    for market_label in market_labels:
        # Get market history
        market_history = market_data[market_label]
        
        # List to store data
        temp_Id = []
        temp_dates = []
        temp_prices = []
        temp_quantity = []
    
        # Update list
        for trade in market_history:
            temp_Id.append(trade['Uuid'])
            temp_dates.append(trade['TimeStamp'])
            temp_prices.append(trade['Price'])
            temp_quantity.append(trade['Quantity'])
    
        # Get unique ids
        unique_ids = list(set(temp_Id))
        unique_ids_index = [temp_Id.index(Id) for Id in unique_ids]
    
        # Get unique transactions
        dates = [temp_dates [ii] for ii in unique_ids_index]
        prices = [temp_prices[ii] for ii in unique_ids_index]
        quantity = [temp_quantity[ii] for ii in unique_ids_index]
    
        # list to dict
        market_history = {'Date': dates,
                          'Price': prices,
                          'Quantity': quantity}
        
        # Dict to df
        market_history = pd.DataFrame(market_history, index = unique_ids_index).sort_values(by = 'Date')
    
        # Save market history
        data_prefix = 'data\\'
        file_name = f'{market_label}' + '.csv'
        market_history.to_csv(data_prefix + file_name, index = False)
        
        # NOTE SAVE THE ID SO THAT WE CAN APPEND NEW TRADES

    return market_labels


def makeTimeBars(market_label, time = '1 hour'):
    """Converts the market history data collected by getUSDMarketData() to 
    time bars. Time bars are build by sampling the data when a predefine number
    of time has passed. 
    
    Parameters
    ----------
    market_history: dict
        Dictionary containing the market history of the last 100 transactions
        
    time: str
        The data will be sampled when a volume number of units have been exchanged
        
    Returns
    ----------
    time_bars: pandas dataframe
        Pandas df containing the volume bars
        
    
    Example
    -------
        None    
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    
    Date: December 12, 2019
    """       
    
    # Convert time into seconds
    if time == '1 min':
        time = 60
    elif time == '5 min':
        time = 5*60
    elif time == '10 min':
        time = 10*60
    elif time == '30 min':
        time = 30*60
    elif time == '1 hour':
        time = 60*60
    elif time == '6 hour':
        time = 6*60*60
    elif time == '12 hour':
        time = 12*60*60
    elif time == '1 day':
        time = 24*60*60
    else:
        time = 5*60
        
    # Set path to data
    data_prefix = 'data\\'
    
    # Set file path
    market_path = data_prefix + f'{market_label}' + '.csv'
    
    # Set columns
    columns = ['Date', 'Price', 'Quantity']
    
    # Read data
    market_history = pd.read_csv(market_path)[columns]
    
    # Get dates, prices, and volumes
    dates = market_history['Date']
    prices = market_history['Price']
    volumes = market_history['Quantity']
    
    # Create list to append bar dates, open, high, low, close, and volume
    bar_dates = []
    bar_open = []
    bar_high = []
    bar_low = []
    bar_close = []
    bar_volume = []
    
    # Preset
    accumulated_time = 0
    lasti = 0
    
    # Make time bars
    for i in range(1, len(prices)):
       
        # Get previous time
        prev_time = parse(dates[i-1])
        
        # Get now time
        now_time = parse(dates[i])
        
        # Get delta time
        delta_time = now_time - prev_time
        delta_time = delta_time.total_seconds()
        
        # Accumulate time
        accumulated_time += delta_time
    
        if accumulated_time >= time:
            # Update list
            bar_dates.append(dates[i])
            bar_open.append(prices[lasti])
            bar_high.append(np.max(prices[lasti:i+1]))
            bar_low.append(np.min(prices[lasti:i+1]))
            bar_close.append(prices[i])
            bar_volume.append(np.sum(volumes[lasti:i+1]))
            
            # Update counter
            lasti = i+1
            
            # Reset time
            accumulated_time = 0
    
    # List to dict
    bars = {'Open': bar_open,
            'High': bar_high,
            'Low': bar_low,
            'Close': bar_close,
            'Volume': bar_volume}
    
    # Create pandas dataframe
    time_bars = pd.DataFrame(bars, index = bar_dates)
    
    return time_bars


def makeTickBars(market_label, frequency = 100):
    """Converts the market history data collected by getUSDMarketData() to 
    tick bars. Volume bars are build by sampling the data when a predefine number
    of trades have been exchanged. The idea for the algorithm was taken
    from Gerard Martínez. 
    
    Parameters
    ----------
    market_history: dict
        Dictionary containing the market history of the last 100 transactions
        
    frequency: float
        The data will be sampled when a frequency number of units have been exchanged
        
    Returns
    ----------
    tick_bars: pandas dataframe
        Pandas df containing the volume bars
        
    
    Example
    -------
        None    
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Gerard Martinez
    Medium: <https://towardsdatascience.com/@gerardmartnez>
    
    Date: November 20, 2019
    """       
    
    
    # Set path to data
    data_prefix = 'data\\'
    
    # Set file path
    market_path = data_prefix + f'{market_label}' + '.csv'
    
    # Set columns
    columns = ['Date', 'Price', 'Quantity']
    
    # Read data
    # Each trade is composed of: [time, price, quantity]
    market_history = pd.read_csv(market_path)[columns]
    
    # Get dates, prices, and volumes
    dates = market_history['Date']
    prices = market_history['Price']
    volumes = market_history['Quantity']
    
    # Create list to append bar dates, open, high, low, close, and volume
    bar_dates = []
    bar_open = []
    bar_high = []
    bar_low = []
    bar_close = []
    bar_volume = []
    
    # Make tick bars
    for i in range(frequency, len(prices), frequency):
        # Update list
        bar_dates.append(dates[i-1] )
        bar_open.append(prices[i-frequency])
        bar_high.append(np.max(prices[i-frequency:i]))
        bar_low.append(np.min(prices[i-frequency:i]))
        bar_close.append(prices[i-1])
        bar_volume.append(np.sum(volumes[i-frequency:i]))
            
    
    # List to dict
    bars = {'Open': bar_open,
            'High': bar_high,
            'Low': bar_low,
            'Close': bar_close,
            'Volume': bar_volume}
    
    # Create pandas dataframe
    tick_bars = pd.DataFrame(bars, index = bar_dates)
    
    return tick_bars

def makeVolumeBars(market_label, volume = 0.25):
    """Converts the market history data collected by getUSDMarketData() to 
    volume bars. Volume bars are build by sampling the data when a predefine number
    of volume have been exchanged. The idea for the algorithm was taken
    from Gerard Martínez. 
    
    Parameters
    ----------
    market_history: dict
        Dictionary containing the market history of the last 100 transactions
        
    volume: float
        The data will be sampled when a volume number of units have been exchanged
        
    Returns
    ----------
    volume_bars: pandas dataframe
        Pandas df containing the volume bars
        
    
    Example
    -------
        None    
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Gerard Martinez
    Medium: <https://towardsdatascience.com/@gerardmartnez>
    
    Date: November 20, 2019
    """       
    
    
    # Set path to data
    data_prefix = 'data\\'
    
    # Set file path
    market_path = data_prefix + f'{market_label}' + '.csv'
    
    # Set columns
    columns = ['Date', 'Price', 'Quantity']
    
    # Read data
    market_history = pd.read_csv(market_path)[columns]
    
    # Get dates, prices, and volumes
    dates = market_history['Date']
    prices = market_history['Price']
    volumes = market_history['Quantity']
    
    # Create list to append bar dates, open, high, low, close, and volume
    bar_dates = []
    bar_open = []
    bar_high = []
    bar_low = []
    bar_close = []
    bar_volume = []
    
    # Preset
    vol = 0
    lasti = 0
    
    for i in range(len(prices)):
        vol += volumes[i]
        if vol >= volume:
            # Update list
            bar_dates.append(dates[i])
            bar_open.append(prices[lasti])
            bar_high.append(np.max(prices[lasti:i+1]))
            bar_low.append(np.min(prices[lasti:i+1]))
            bar_close.append(prices[i])
            bar_volume.append(np.sum(volumes[lasti:i+1]))
            
            # Update counter
            lasti = i+1
            
            # Reset volume
            vol = 0
    
    # List to dict
    bars = {'Open': bar_open,
            'High': bar_high,
            'Low': bar_low,
            'Close': bar_close,
            'Volume': bar_volume}
    
    # Create pandas dataframe
    volume_bars = pd.DataFrame(bars, index = bar_dates)
    
    return volume_bars


def makeDollarBars(market_label, dollars = 10):
    """Converts the market history data collected by getUSDMarketData() to 
    volume bars. Dollar bars are build by sampling the data when a predefine number
    of dollars have been exchanged. The idea for the algorithm was taken
    from Gerard Martínez. 
    
    Parameters
    ----------
    market_history: dict
        Dictionary containing the market history of the last 100 transactions
        
    dollars: float
        The data will be sampled when a dollar number of units have been exchanged
        
    Returns
    ----------
    dollar_bars: pandas dataframe
        Pandas df containing the volume bars
        
    
    Example
    -------
        None    
        
    Author Information
    ------------------
    Frank Ceballos
    LinkedIn: <https://www.linkedin.com/in/frank-ceballos/>
    
    Gerard Martinez
    Medium: <https://towardsdatascience.com/@gerardmartnez>
    
    Date: December 11, 2019
    """       
    
    
    # Set path to data
    data_prefix = 'data\\'
    
    # Set file path
    market_path = data_prefix + f'{market_label}' + '.csv'
    
    # Set columns
    columns = ['Date', 'Price', 'Quantity']
    
    # Read data
    market_history = pd.read_csv(market_path)[columns]
    
    # Get dates, prices, and volumes
    dates = market_history['Date']
    prices = market_history['Price']
    volumes = market_history['Quantity']
    
    # Create list to append bar dates, open, high, low, close, and volume
    bar_dates = []
    bar_open = []
    bar_high = []
    bar_low = []
    bar_close = []
    bar_volume = []
    
    # Preset
    dol = 0
    lasti = 0
    
    for i in range(len(prices)):
        dol += volumes[i]*prices[i]
        if dol >= dollars:
            # Update list
            bar_dates.append(dates[i])
            bar_open.append(prices[lasti])
            bar_high.append(np.max(prices[lasti:i+1]))
            bar_low.append(np.min(prices[lasti:i+1]))
            bar_close.append(prices[i])
            bar_volume.append(np.sum(volumes[lasti:i+1]))
            
            # Update counter
            lasti = i+1
            
            # Reset dollars
            dol = 0
    
    # List to dict
    bars = {'Open': bar_open,
            'High': bar_high,
            'Low': bar_low,
            'Close': bar_close,
            'Volume': bar_volume}
    
    # Create pandas dataframe
    dollar_bars = pd.DataFrame(bars, index = bar_dates)
    
    return dollar_bars 
