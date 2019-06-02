# Libraries
import pandas as pd
import numpy as np
import fbprophet
from pytrends.request import TrendReq
import pandas_datareader as wb
import matplotlib.pyplot as plt
import matplotlib


# Main class for data analysis
class Stocker():

    # Initialization requires a ticker symbol
    def __init__(self, ticker):

        # Enforce capitalization
        ticker = ticker.upper()

        # Symbol is used for labeling plots
        self.symbol = ticker

        # Retrieval the financial data
        # CIMB = "1023.KL"
        try:
            stock = wb.get_data_yahoo (ticker, start='2017-1-1', end='2018-1-1')

        except Exception as e:
            print('Error Retrieving Data.')
            print(e)
            return

        # Set the index to a column called Date
        stock = stock.reset_index(level=0)

        # Columns required for prophet
        stock['ds'] = stock['Date']
        stock['y'] = stock['Adj Close']
        stock['Daily Change'] = stock['Adj Close'] - stock['Open']

        # Data assigned as class attribute
        self.stock = stock.copy()

        # Minimum and maximum date in range
        self.min_date = min(stock['Date'])
        self.max_date = max(stock['Date'])

        # Find max and min prices and dates on which they occurred
        self.max_price = np.max(self.stock['y'])
        self.min_price = np.min(self.stock['y'])

        self.min_price_date = self.stock[self.stock['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.stock[self.stock['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]

        # The starting price (starting with the opening price)
        self.starting_price = float(self.stock.ix[0, 'Open'])

        # The most recent price
        self.most_recent_price = float(self.stock.ix[len(self.stock) - 1, 'y'])

        # Whether or not to round dates
        self.round_dates = True

        # Number of years of data to train on
        self.training_years = 1

        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None

        print('{} Stocker Initialized. Data covers {} to {}.'.format(self.symbol,
                                                                     self.min_date.date(),
                                                                     self.max_date.date()))

    """
        Make sure start and end dates are in the range and can be
        converted to pandas datetimes. Returns dates in the correct format
        """

    def handle_dates(self, start_date, end_date):

        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date

        try:
            # Convert to pandas datetime for indexing dataframe
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

        except Exception as e:
            print('Enter valid pandas date format.')
            print(e)
            return

        valid_start = False
        valid_end = False

        # User will continue to enter dates until valid dates are met
        while (not valid_start) & (not valid_end):
            valid_end = True
            valid_start = True

            if end_date.date() < start_date.date():
                print('End Date must be later than start date.')
                start_date = pd.to_datetime(input('Enter a new start date: '))
                end_date = pd.to_datetime(input('Enter a new end date: '))
                valid_end = False
                valid_start = False

            else:
                if end_date.date() > self.max_date.date():
                    print('End Date exceeds data range')
                    end_date = pd.to_datetime(input('Enter a new end date: '))
                    valid_end = False

                if start_date.date() < self.min_date.date():
                    print('Start Date is before date range')
                    start_date = pd.to_datetime(input('Enter a new start date: '))
                    valid_start = False

        return start_date, end_date

    """
        Return the dataframe trimmed to the specified range.
    """

    def make_df(self, start_date, end_date, df=None):

        # Default is to use the object stock data
        if not df:
            df = self.stock.copy()

        start_date, end_date = self.handle_dates(start_date, end_date)

        # keep track of whether the start and end dates are in the data
        start_in = True
        end_in = True

        # If user wants to round dates (default behavior)
        if self.round_dates:
            # Record if start and end date are in df
            if (start_date not in list(df['Date'])):
                start_in = False
            if (end_date not in list(df['Date'])):
                end_in = False

            # If both are not in dataframe, round both
            if (not end_in) & (not start_in):
                trim_df = df[(df['Date'] >= start_date.date()) &
                             (df['Date'] <= end_date.date())]

            else:
                # If both are in dataframe, round neither
                if (end_in) & (start_in):
                    trim_df = df[(df['Date'] >= start_date.date()) &
                                 (df['Date'] <= end_date.date())]
                else:
                    # If only start is missing, round start
                    if (not start_in):
                        trim_df = df[(df['Date'] > start_date.date()) &
                                     (df['Date'] <= end_date.date())]
                    # If only end is imssing round end
                    elif (not end_in):
                        trim_df = df[(df['Date'] >= start_date.date()) &
                                     (df['Date'] < end_date.date())]


        else:
            valid_start = False
            valid_end = False
            while (not valid_start) & (not valid_end):
                start_date, end_date = self.handle_dates(start_date, end_date)

                # No round dates, if either data not in, print message and return
                if (start_date in list(df['Date'])):
                    valid_start = True
                if (end_date in list(df['Date'])):
                    valid_end = True

                # Check to make sure dates are in the data
                if (start_date not in list(df['Date'])):
                    print('Start Date not in data (either out of range or not a trading day.)')
                    start_date = pd.to_datetime(input(prompt='Enter a new start date: '))

                elif (end_date not in list(df['Date'])):
                    print('End Date not in data (either out of range or not a trading day.)')
                    end_date = pd.to_datetime(input(prompt='Enter a new end date: '))

            # Dates are not rounded
            trim_df = df[(df['Date'] >= start_date.date()) &
                         (df['Date'] <= end_date.date())]

        return trim_df

    def reset_plot(self):

        # Restore default parameters
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        # Adjust a few parameters to liking
        matplotlib.rcParams['figure.figsize'] = (8, 5)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'

    # Method to linearly interpolate prices on the weekends
    def resample(self, dataframe):
        # Change the index and resample at daily level
        dataframe = dataframe.set_index('ds')
        dataframe = dataframe.resample('D')

        # Reset the index and interpolate nan values
        dataframe = dataframe.reset_index(level=0)
        dataframe = dataframe.interpolate()
        return dataframe

    # Remove weekends from a dataframe
    def remove_weekends(self, dataframe):

        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)

        weekends = []

        # Find all of the weekends
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)

        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)

        return dataframe

    # Basic Historical Plots and Basic Statistics
    def plot_stock(self, start_date=None, end_date=None, stats=['Adj Close'], plot_type='basic'):

        self.reset_plot()

        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date

        stock_plot = self.make_df(start_date, end_date)

        colors = ['r', 'b', 'g', 'y', 'c', 'm']

        for i, stat in enumerate(stats):

            stat_min = min(stock_plot[stat])
            stat_max = max(stock_plot[stat])

            stat_avg = np.mean(stock_plot[stat])

            date_stat_min = stock_plot[stock_plot[stat] == stat_min]['Date']
            date_stat_min = date_stat_min[date_stat_min.index[0]].date()
            date_stat_max = stock_plot[stock_plot[stat] == stat_max]['Date']
            date_stat_max = date_stat_max[date_stat_max.index[0]].date()

            print('Maximum {} = {:.2f} on {}.'.format(stat, stat_max, date_stat_max))
            print('Minimum {} = {:.2f} on {}.'.format(stat, stat_min, date_stat_min))
            print('Current {} = {:.2f} on {}.\n'.format(stat, self.stock.ix[len(self.stock) - 1, stat],
                                                        self.max_date.date()))

            # Percentage y-axis
            if plot_type == 'pct':
                # Simple Plot
                plt.style.use('fivethirtyeight');
                if stat == 'Daily Change':
                    plt.plot(stock_plot['Date'], 100 * stock_plot[stat],
                             color=colors[i], linewidth=2.4, alpha=0.9,
                             label=stat)
                else:
                    plt.plot(stock_plot['Date'], 100 * (stock_plot[stat] - stat_avg) / stat_avg,
                             color=colors[i], linewidth=2.4, alpha=0.9,
                             label=stat)

                plt.xlabel('Date');
                plt.ylabel('Change Relative to Average (%)');
                plt.title('%s Stock History' % self.symbol);
                plt.legend(prop={'size': 10})
                plt.grid(color='k', alpha=0.4);

                # Stat y-axis
            elif plot_type == 'basic':
                plt.style.use('fivethirtyeight');
                plt.plot(stock_plot['Date'], stock_plot[stat], color=colors[i], linewidth=3, label=stat, alpha=0.8)
                plt.xlabel('Date');
                plt.ylabel('RM');
                plt.title('%s Stock History' % self.symbol);
                plt.legend(prop={'size': 10})
                plt.grid(color='k', alpha=0.4);

        plt.show();

    def retrieve_google_trends(self, search, date_range):

        # Set up the trend fetching object
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [search]

        try:

            # Create the search object
            pytrends.build_payload(kw_list, cat=0, timeframe=date_range[0], geo='', gprop='news')

            # Retrieve the interest over time
            trends = pytrends.interest_over_time()

            related_queries = pytrends.related_queries()

        except Exception as e:
            print('\nGoogle Search Trend retrieval failed.')
            print(e)
            return

        return trends, related_queries

    # Create a prophet model without training
    def create_model(self):

        # Make the model
        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,
                                      weekly_seasonality=self.weekly_seasonality,
                                      yearly_seasonality=self.yearly_seasonality,
                                      changepoint_prior_scale=self.changepoint_prior_scale,
                                      changepoints=self.changepoints)

        if self.monthly_seasonality:
                # Add monthly seasonality
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        return model

    def change_analysis(self, count):
        self.reset_plot()

        model = self.create_model()
        train = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years=self.training_years)).date()]
        # train = self.stock
        model.fit(train)

        # Predictions of the training data (no future periods)
        future = model.make_future_dataframe(periods=0, freq='D')
        future = model.predict(future)

        train = pd.merge(train, future[['ds', 'yhat']], on='ds', how='inner')

        train.to_csv('train_cimb.csv')

        changepoints = model.changepoints
        train = train.reset_index(drop=True)

        # Create dataframe of only changepoints
        change_indices = []
        for changepoint in (changepoints):
            change_indices.append(train[train['ds'] == changepoint.date()].index[0])

        c_data = train.loc[change_indices, :]
        deltas = model.params['delta'][0]

        c_data['delta'] = deltas
        c_data['abs_delta'] = abs(c_data['delta'])

        # c_data.to_csv('cimb_changes.csv')

        # Sort the values by maximum change
        c_data = c_data.sort_values(by='abs_delta', ascending=False)

        # Print changes
        print(c_data[:count])

    def changepoint_date_analysis(self, search=None):
        self.reset_plot()

        model = self.create_model()

        # Use past self.training_years years of data
        train = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years=self.training_years)).date()]
        # train = self.stock
        model.fit(train)

        # Predictions of the training data (no future periods)
        future = model.make_future_dataframe(periods=0, freq='D')
        future = model.predict(future)

        train = pd.merge(train, future[['ds', 'yhat']], on='ds', how='inner')

        changepoints = model.changepoints
        train = train.reset_index(drop=True)

        # Create dataframe of only changepoints
        change_indices = []
        for changepoint in (changepoints):
            change_indices.append(train[train['ds'] == changepoint.date()].index[0])

        c_data = train.ix[change_indices, :]
        deltas = model.params['delta'][0]

        c_data['delta'] = deltas
        c_data['abs_delta'] = abs(c_data['delta'])


        # Sort the values by maximum change
        c_data = c_data.sort_values(by='abs_delta', ascending=False)

        # Limit to 10 largest changepoints
        c_data = c_data[:10]

        # Separate into negative and positive changepoints
        cpos_data = c_data[c_data['delta'] > 0]
        cneg_data = c_data[c_data['delta'] < 0]

        # Changepoints and data
        if not search:
            print('\nChangepoints sorted by slope rate of change (2nd derivative):\n')
            print(c_data.ix[:, ['Date', 'Close', 'delta']][:5])

            # Line plot showing actual values, estimated values, and changepoints
            self.reset_plot()

            # Set up line plot
            plt.plot(train['ds'], train['y'], 'ko', ms=4, label='Stock Price')
            plt.plot(future['ds'], future['yhat'], color='navy', linewidth=2.0, label='Modeled')


            # Changepoints as vertical lines
            plt.vlines(cpos_data['ds'].dt.to_pydatetime(), ymin=min(train['y']), ymax=max(train['y']),
                       linestyles='dashed', color='r',
                       linewidth=1.2, label='Negative Changepoints')

            plt.vlines(cneg_data['ds'].dt.to_pydatetime(), ymin=min(train['y']), ymax=max(train['y']),
                       linestyles='dashed', color='darkgreen',
                       linewidth=1.2, label='Positive Changepoints')

            plt.legend(prop={'size': 10});
            plt.xlabel('Date');
            plt.ylabel('Price (RM)');
            plt.title('Stock Price with Changepoints')
            plt.show()

        # Search for search term in google news
        # Show related queries, rising related queries
        # Graph changepoints, search frequency, stock price
        if search:
            date_range = ['%s %s' % (str(min(train['Date']).date()), str(max(train['Date']).date()))]

            # Get the Google Trends for specified terms and join to training dataframe
            trends, related_queries = self.retrieve_google_trends(search, date_range)

            if (trends is None) or (related_queries is None):
                print('No search trends found for %s' % search)
                return

            print('\n Top Related Queries: \n')
            print(related_queries[search]['top'].head())
            print(related_queries[search])

            print('\n Rising Related Queries: \n')
            print(related_queries[search]['rising'].head())

            # Upsample the data for joining with training data
            trends = trends.resample('D')
            # print("trends intrrpolate")
            # print(trends.interpolate())

            # Interpolate the frequency
            trends = trends.interpolate()

            # trends.reset_index(level=0)

            trends = trends.reset_index()
            trends.reset_index(inplace=True)
            trends = trends.rename(columns={'date': 'ds', search: 'freq'})

            # Merge with the training data
            train = pd.merge(train, trends, on='ds', how='inner')

            # Normalize values
            train['y_norm'] = train['y'] / max(train['y'])
            train['freq_norm'] = train['freq'] / max(train['freq'])

            self.reset_plot()

            # Plot the normalized stock price and normalize search frequency
            plt.plot(train['ds'], train['y_norm'], 'k-', label='Stock Price')
            plt.plot(train['ds'], train['freq_norm'], color='goldenrod', label='Search Frequency')

            # Changepoints as vertical lines
            plt.vlines(cpos_data['ds'].dt.to_pydatetime(), ymin=0, ymax=1,
                       linestyles='dashed', color='r',
                       linewidth=1.2, label='Negative Changepoints')

            plt.vlines(cneg_data['ds'].dt.to_pydatetime(), ymin=0, ymax=1,
                       linestyles='dashed', color='darkgreen',
                       linewidth=1.2, label='Positive Changepoints')

            # Plot formatting
            plt.legend(prop={'size': 10})
            plt.xlabel('Date');
            plt.ylabel('Normalized Values');
            plt.title('%s Stock Price and Search Frequency for %s' % (self.symbol, search))
            plt.show()
            # train.to_csv('train_analysis.csv')

    def changepoint_news_analysis(self):
        self.reset_plot()

        model = self.create_model()

        # Use past self.training_years years of data
        train = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years=self.training_years)).date()]
        # train = self.stock
        model.fit(train)

        # Predictions of the training data (no future periods)
        future = model.make_future_dataframe(periods=0, freq='D')
        future = model.predict(future)

        train = pd.merge(train, future[['ds', 'yhat']], on='ds', how='inner')
        train = train.reset_index(drop=True)

        # Line plot showing actual values, estimated values, and changepoints
        self.reset_plot()

        # Import news
        news = pd.read_csv('cimb_news_cimb_news.csv')
        print(news.head(10))

        train = pd.read_csv('plot_news.csv')
        # print(train.head(20))
        # seperate into 1 and 0 news
        cpos_data = news[news['Score'] == 1]
        cneu_data = news[news['Score'] == 0]

        plt.plot(train['ds'], train['y'], 'ko', ms=4, label='Stock Price')
        plt.plot(train['ds'], train['yhat'], color='navy', linewidth=2.0, label='Modeled')
        # plt.show()

        # df["TimeReviewed"] = pd.to_datetime(df["TimeReviewed"])
        cpos_data['Date'] = pd.to_datetime(cpos_data['Date'])
        cneu_data['Date'] = pd.to_datetime(cneu_data['Date'])

        # Changepoints as vertical lines
        plt.vlines(cpos_data["Date"].astype(str), ymin=min(train['y']), ymax=max(train['y']),
                   linestyles='dashed', color='r',
                   linewidth=1.2, label='Positive News')

        # plt.vlines(cneu_data['Date'].astype(str), ymin=min(train['y']), ymax=max(train['y']),
        #            linestyles='dashed', color='darkgreen',
        #            linewidth=1.2, label='Neutral News')

        plt.legend(prop={'size': 10});
        plt.xlabel('Date');
        plt.ylabel('Price (RM)');
        plt.title('Stock Price with News')
        plt.show()


# test Stocker Class
CIMB = "1023.KL"
cimb = Stocker(CIMB)
#
cimb.plot_stock()
cimb.plot_stock(stats = ['Daily Change', 'Volume'],  plot_type='pct')
cimb.change_analysis(10)
cimb.changepoint_date_analysis()
cimb.changepoint_date_analysis(search = 'US China')
cimb.changepoint_date_analysis(search = 'Donald Trump')
# cimb.changepoint_news_analysis()

# sunway
SUNWAY = "5211.KL"
sunway = Stocker(SUNWAY)
sunway.plot_stock()
sunway.changepoint_date_analysis(search = 'US China')

# OSK
OSK = "5053.KL"
osk = Stocker(OSK)
osk.plot_stock()

# Airport
AIRPORT = "5014.KL"
airport = Stocker(AIRPORT)
airport.plot_stock()