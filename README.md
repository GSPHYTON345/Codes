# Grafico ottimizzato
import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime, time
from google.colab import drive
import optuna
from sklearn.model_selection import TimeSeriesSplit

drive.mount('/content/drive')

# Define the path to your CSV file in Google Drive
file_path = '/content/drive/My Drive/GBEBROKERS_EURUSD, 5 MARZO.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Select only the required columns
data = data[['time', 'open', 'high', 'low', 'close']]

# Convert the 'time' column to datetime
data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
data.dropna(subset=['time'], inplace=True)
data.set_index('time', inplace=True)

def calculate_supertrend(data, atr_period, multiplier):
    data['ATR'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=atr_period)
    data['hl2'] = (data['high'] + data['low']) / 2
    data['Upper_Band'] = data['hl2'] - (multiplier * data['ATR'])
    data['Lower_Band'] = data['hl2'] + (multiplier * data['ATR'])
    data['Supertrend'] = np.nan
    data['Trend'] = np.nan

    for i in range(1, len(data)):
        if np.isnan(data['Trend'].iloc[i-1]):
            data.loc[data.index[i], 'Trend'] = 1
        elif data['close'].iloc[i] > data['Lower_Band'].iloc[i-1]:
            data.loc[data.index[i], 'Trend'] = 1
        elif data['close'].iloc[i] < data['Upper_Band'].iloc[i-1]:
            data.loc[data.index[i], 'Trend'] = -1
        else:
            data.loc[data.index[i], 'Trend'] = data['Trend'].iloc[i-1]

        if data['Trend'].iloc[i] == 1:
            data.loc[data.index[i], 'Upper_Band'] = max(data['Upper_Band'].iloc[i], data['Upper_Band'].iloc[i-1])
        else:
            data.loc[data.index[i], 'Lower_Band'] = min(data['Lower_Band'].iloc[i], data['Lower_Band'].iloc[i-1])

        supertrend_value = data['Upper_Band'].iloc[i] if data['Trend'].iloc[i] == 1 else data['Lower_Band'].iloc[i]
        data.loc[data.index[i], 'Supertrend'] = round(supertrend_value, 5)

    return data

def run_backtest(data, atr_period, multiplier, fast_length, slow_length, signal_length):
    if fast_length >= slow_length or fast_length <= 0 or slow_length <= 0:
        return 0, 0, 0, 0

    data = calculate_supertrend(data, atr_period, multiplier)

    try:
        data['MACD'], data['Signal'], data['Hist'] = ta.MACD(data['close'], fastperiod=fast_length, slowperiod=slow_length, signalperiod=signal_length)
    except Exception as e:
        return 0, 0, 0, 0

    data['MACD'] = data['MACD'].round(5)
    data['Signal'] = data['Signal'].round(5)
    data['Hist'] = data['Hist'].round(5)

    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)

    def within_time_range(timestamp):
        return time(6, 30) <= timestamp.time() <= time(19, 30)

    data['WithinTime'] = data.index.map(within_time_range)

    data['LongCondition'] = (data['Trend'] == 1) & (data['MACD'] > data['Signal']) & (data['close'] > data['Supertrend']) & (data['Trend'].shift(1) == -1) & data['WithinTime']
    data['ShortCondition'] = (data['Trend'] == -1) & (data['MACD'] < data['Signal']) & (data['close'] < data['Supertrend']) & (data['Trend'].shift(1) == 1) & data['WithinTime']

    initial_equity = 100000
    equity = initial_equity

    def calculate_position_size(equity, entry_price, stop_loss_price):
        risk_per_trade = equity * 0.01
        risk_pips = abs(entry_price - stop_loss_price)
        qty = risk_per_trade / risk_pips
        return qty

    trades = []
    equity_curve = [equity]
    current_trades = []

    for i in range(1, len(data)):
        if data['LongCondition'].iloc[i] or data['ShortCondition'].iloc[i]:
            if data['LongCondition'].iloc[i]:
                entry_price = data['close'].iloc[i]
                stop_loss = data['Supertrend'].iloc[i]
                take_profit = entry_price + (abs(entry_price - stop_loss) * 2)
                qty = calculate_position_size(equity, entry_price, stop_loss)
                new_trade = {
                    'Type': 'Long',
                    'EntryTime': data.index[i],
                    'EntryPrice': entry_price,
                    'StopLoss': stop_loss,
                    'TakeProfit': take_profit,
                    'Qty': qty,
                    'ExitPrice': np.nan,
                    'ExitTime': np.nan,
                    'Profit': np.nan,
                    'Outcome': 'Open'
                }
                current_trades.append(new_trade)
            elif data['ShortCondition'].iloc[i]:
                entry_price = data['close'].iloc[i]
                stop_loss = data['Supertrend'].iloc[i]
                take_profit = entry_price - (abs(entry_price - stop_loss) * 2)
                qty = calculate_position_size(equity, entry_price, stop_loss)
                new_trade = {
                    'Type': 'Short',
                    'EntryTime': data.index[i],
                    'EntryPrice': entry_price,
                    'StopLoss': stop_loss,
                    'TakeProfit': take_profit,
                    'Qty': qty,
                    'ExitPrice': np.nan,
                    'ExitTime': np.nan,
                    'Profit': np.nan,
                    'Outcome': 'Open'
                }
                current_trades.append(new_trade)

        for trade in current_trades:
            if trade['Outcome'] == 'Open':
                if trade['Type'] == 'Long' and data['ShortCondition'].iloc[i]:
                    trade['Outcome'] = 'Closed for Reverse'
                    trade['ExitPrice'] = data['close'].iloc[i]
                    trade['ExitTime'] = data.index[i]
                    trade['Profit'] = (trade['ExitPrice'] - trade['EntryPrice']) * trade['Qty']
                    equity += trade['Profit']
                    trades.append(trade)
                    if not any(t['Type'] == 'Short' and t['Outcome'] == 'Open' for t in current_trades):
                        entry_price = data['close'].iloc[i]
                        stop_loss = data['Supertrend'].iloc[i]
                        take_profit = entry_price - (abs(entry_price - stop_loss) * 2)
                        qty = calculate_position_size(equity, entry_price, stop_loss)
                        reverse_trade = {
                            'Type': 'Short',
                            'EntryTime': data.index[i],
                            'EntryPrice': entry_price,
                            'StopLoss': stop_loss,
                            'TakeProfit': take_profit,
                            'Qty': qty,
                            'ExitPrice': np.nan,
                            'ExitTime': np.nan,
                            'Profit': np.nan,
                            'Outcome': 'Open'
                        }
                        current_trades.append(reverse_trade)
                elif trade['Type'] == 'Short' and data['LongCondition'].iloc[i]:
                    trade['Outcome'] = 'Closed for Reverse'
                    trade['ExitPrice'] = data['close'].iloc[i]
                    trade['ExitTime'] = data.index[i]
                    trade['Profit'] = (trade['EntryPrice'] - trade['ExitPrice']) * trade['Qty']
                    equity += trade['Profit']
                    trades.append(trade)
                    if not any(t['Type'] == 'Long' and t['Outcome'] == 'Open' for t in current_trades):
                        entry_price = data['close'].iloc[i]
                        stop_loss = data['Supertrend'].iloc[i]
                        take_profit = entry_price + (abs(entry_price - stop_loss) * 2)
                        qty = calculate_position_size(equity, entry_price, stop_loss)
                        reverse_trade = {
                            'Type': 'Long',
                            'EntryTime': data.index[i],
                            'EntryPrice': entry_price,
                            'StopLoss': stop_loss,
                            'TakeProfit': take_profit,
                            'Qty': qty,
                            'ExitPrice': np.nan,
                            'ExitTime': np.nan,
                            'Profit': np.nan,
                            'Outcome': 'Open'
                        }
                        current_trades.append(reverse_trade)
                elif trade['Type'] == 'Long' and data['low'].iloc[i] < trade['StopLoss']:
                    trade['Outcome'] = 'Loss'
                    trade['ExitPrice'] = trade['StopLoss']
                    trade['ExitTime'] = data.index[i]
                    trade['Profit'] = (trade['ExitPrice'] - trade['EntryPrice']) * trade['Qty']
                    equity += trade['Profit']
                    trades.append(trade)
                elif trade['Type'] == 'Long' and data['high'].iloc[i] > trade['TakeProfit']:
                    trade['Outcome'] = 'Win'
                    trade['ExitPrice'] = trade['TakeProfit']
                    trade['ExitTime'] = data.index[i]
                    trade['Profit'] = (trade['ExitPrice'] - trade['EntryPrice']) * trade['Qty']
                    equity += trade['Profit']
                    trades.append(trade)
                elif trade['Type'] == 'Short' and data['high'].iloc[i] > trade['StopLoss']:
                    trade['Outcome'] = 'Loss'
                    trade['ExitPrice'] = trade['StopLoss']
                    trade['ExitTime'] = data.index[i]
                    trade['Profit'] = (trade['EntryPrice'] - trade['ExitPrice']) * trade['Qty']
                    equity += trade['Profit']
                    trades.append(trade)
                elif trade['Type'] == 'Short' and data['low'].iloc[i] < trade['TakeProfit']:
                    trade['Outcome'] = 'Win'
                    trade['ExitPrice'] = trade['TakeProfit']
                    trade['ExitTime'] = data.index[i]
                    trade['Profit'] = (trade['EntryPrice'] - trade['ExitPrice']) * trade['Qty']
                    equity += trade['Profit']
                    trades.append(trade)

        current_trades = [trade for trade in current_trades if trade['Outcome'] == 'Open']
        equity_curve.append(equity)

    trades_df = pd.DataFrame(trades)
    win_rate = len(trades_df[trades_df['Outcome'] == 'Win']) / len(trades_df) if len(trades_df) > 0 else 0
    equity_series = pd.Series(equity_curve)
    drawdown = equity_series / equity_series.cummax() - 1
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

    sharpe_ratio = (equity_series.pct_change().mean() / equity_series.pct_change().std()) * np.sqrt(252)

    return equity, win_rate, max_drawdown, sharpe_ratio

# Time Series Split for Cross Validation
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    atr_period = trial.suggest_int('atr_period', 5, 25)
    multiplier = trial.suggest_uniform('multiplier', 1.5, 4.0)
    fast_length = trial.suggest_int('fast_length', 5, 150)
    slow_length = trial.suggest_int('slow_length', 10, 150)
    signal_length = trial.suggest_int('signal_length', 5, 115)

    cv_results = []
    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        final_equity, win_rate, max_drawdown, sharpe_ratio = run_backtest(train_data, atr_period, multiplier, fast_length, slow_length, signal_length)
        cv_results.append(sharpe_ratio)
    return -np.mean(cv_results)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

best_params = study.best_params
print(f"Refined Best parameters: {best_params}")

# Collect all results
results = []

for trial in study.trials:
    params = trial.params
    final_equity, win_rate, max_drawdown, sharpe_ratio = run_backtest(data.copy(), params['atr_period'], params['multiplier'], params['fast_length'], params['slow_length'], params['signal_length'])
    results.append((params, final_equity, win_rate, max_drawdown, sharpe_ratio))

results_df = pd.DataFrame(results, columns=['params', 'final_equity', 'win_rate', 'max_drawdown', 'sharpe_ratio'])

# Print the 10 best combinations for net profit
print("Top 10 Combinations by Net Profit:")
print(results_df.nlargest(10, 'final_equity'))

# Print the 10 best combinations for win rate
print("Top 10 Combinations by Win Rate:")
print(results_df.nlargest(10, 'win_rate'))

# Print the 10 best combinations for drawdown
print("Top 10 Combinations by Lowest Drawdown:")
print(results_df.nsmallest(10, 'max_drawdown'))

# Print the 10 best combinations for Sharpe Ratio
print("Top 10 Combinations by Sharpe Ratio:")
print(results_df.nlargest(10, 'sharpe_ratio'))
