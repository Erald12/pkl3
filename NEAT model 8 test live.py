# Import necessary libraries
import ta
from tvDatafeed import TvDatafeed, Interval
import numpy as np
import pandas as pd
import time
from scipy.stats import linregress
import neat
import pickle
from rich.console import Console
from datetime import datetime, timedelta
from datetime import datetime


# Fetch data with retry logic
def fetch_data_with_retry(tv, symbol, exchange, interval, n_bars, retries=3):
    """Fetch historical data with retry logic."""
    for attempt in range(retries):
        try:
            df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
            if df is not None and not df.empty:
                return df
            else:
                print(f"No data returned for {symbol}. Retrying... (Attempt {attempt + 1}/{retries})")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
        time.sleep(2)  # Wait before retrying
    return None  # Return None if all retries fail


# Function to calculate slope over a rolling window
def calculate_slope(series, window=7):
    slopes = [np.nan] * (window - 1)  # Fill initial rows with NaN
    for i in range(window - 1, len(series)):
        y = series[i - window + 1:i + 1]
        x = np.arange(window)
        # Perform linear regression and extract the slope
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)



# Define a trading environment with a feedback loop
class TradingEnvironment:
    def __init__(self, data, starting_balance=20):
        self.data = data
        self.balance = starting_balance
        self.current_step = 0
        self.winloss = []
        self.equity_history = [starting_balance]
        self.profits = [float(starting_balance)]
        self.close_price = data['close_symbolused'].values
        self.open_price = data['open_symbolused'].values
        self.high_price = data['high_symbolused'].values
        self.low_price = data['low_symbolused'].values
        self.ema5 = data['ema5'].values
        self.ema8 = data['ema8'].values
        self.ema100 = data['ema100'].values

    def reset(self):
        self.balance = 20
        self.current_step = 0
        self.equity_history = [self.balance]

    def step(self, action):
        # Actions: 0 = hold, 1 = buy, -1 = sell
        price = self.close_price[self.current_step]
        low_and_high = []

        # Look ahead for the next 100 steps to store highs and lows
        for step_ahead in range(1, 400):
            if self.current_step + step_ahead < len(self.close_price):
                future_low = self.low_price[self.current_step + step_ahead]
                future_high = self.high_price[self.current_step + step_ahead]
                low_and_high.append((future_low, future_high))

        ema5_current, ema8_current, ema100_current = self.ema5[self.current_step], self.ema8[self.current_step], self.ema100[self.current_step]
        ema5_prev, ema8_prev, ema100_prev = self.ema5[self.current_step - 1], self.ema8[self.current_step - 1], self.ema100[self.current_step - 1]

        if action == 1 and ema5_prev > ema100_prev and ema8_prev < ema100_prev and ema8_current> ema100_current and ema5_current>ema100_current and price > self.open_price[self.current_step]:
            loss_pct = 0.03
            gain_pct = 0.15
            stop_loss = ema100_current
            tp_limit = ((gain_pct * price) / abs(-loss_pct / ((stop_loss - price) / price))) + price

            # Evaluate if TP or SL is hit in the lookahead prices
            for i in low_and_high:
                low, high = i
                if high >= tp_limit:  # Take Profit hit1
                    self.balance += self.balance * gain_pct
                    self.winloss.append(1)
                    self.profits.append(sum(self.profits)*gain_pct)
                    break
                elif low <= stop_loss:  # Stop Loss hit
                    self.balance -= self.balance * loss_pct
                    self.profits.append(sum(self.profits) * loss_pct)
                    self.winloss.append(-1)
                    break
        elif action == -1 and ema5_prev < ema100_prev and ema8_prev > ema100_prev and ema8_current< ema100_current and ema5_current<ema100_current and price < self.open_price[
            self.current_step]:
            loss_pct = 0.03
            gain_pct = 0.15
            stop_loss = ema100_current
            tp_limit = ((-gain_pct * price) / abs(loss_pct / ((stop_loss - price) / price))) + price
            for i in low_and_high:
                low, high = i
                if low <= tp_limit:  # Take Profit hit for sell
                    self.balance += self.balance * gain_pct
                    self.profits.append(sum(self.profits) * gain_pct)
                    self.winloss.append(1)
                    break
                elif high >= stop_loss:  # Stop Loss hit for sell
                    self.balance -= self.balance * loss_pct
                    self.profits.append(sum(self.profits) * loss_pct)
                    self.winloss.append(-1)
                    break

        self.current_step += 1
        self.equity_history.append(self.balance)
        done = self.current_step >= len(self.data)
        return self.balance, done, self.winloss, sum(self.profits)


def create_neat_config():
    config_content = """
    [NEAT]
    pop_size = 100
    fitness_criterion = max
    fitness_threshold = 999999999999999999999
    reset_on_extinction = True

    [DefaultGenome]
    feed_forward = False

    # Node activation functions
    activation_default = sigmoid
    activation_mutate_rate = 0.0
    activation_options = sigmoid softmax hard_sigmoid scaled_sigmoid logistic tanh relu

    # Node aggregation functions
    aggregation_default = sum
    aggregation_mutate_rate = 0.0
    aggregation_options = sum mean product

    # Structural mutation rates
    single_structural_mutation = False
    structural_mutation_surer = 0
    conn_add_prob = 0.5
    conn_delete_prob = 0.2
    node_add_prob = 0.2
    node_delete_prob = 0.2

    # Connection parameters
    initial_connection = full_direct
    bias_init_mean = 0.0
    bias_init_stdev = 1.0
    bias_max_value = 30.0
    bias_min_value = -30.0
    bias_mutate_power = 0.5
    bias_mutate_rate = 0.1
    bias_replace_rate = 0.1

    # Response parameters (added these)
    response_init_mean = 0.0
    response_init_stdev = 1.0
    response_replace_rate = 0.1
    response_mutate_rate = 0.1
    response_mutate_power = 0.5
    response_max_value = 30.0
    response_min_value = -30.0

    # Default enabled state
    enabled_default = True

    # Enable mutation rate
    enabled_mutate_rate = 0.1

    # Node parameters
    num_hidden = 0
    num_inputs = 12
    num_outputs = 3

    # Connection mutation
    weight_init_mean = 0.0
    weight_init_stdev = 1.0
    weight_max_value = 30
    weight_min_value = -30
    weight_mutate_power = 0.5
    weight_mutate_rate = 0.8
    weight_replace_rate = 0.1

    # Compatibility parameters
    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient = 0.5

    [DefaultSpeciesSet]
    compatibility_threshold = 3.0

    [DefaultStagnation]
    species_fitness_func = max
    max_stagnation = 15
    species_elitism = 2

    [DefaultReproduction]
    elitism = 2
    survival_threshold = 0.2
    """
    with open('neat_config5.txt', 'w') as f:
        f.write(config_content)


# Function to test the trained NEAT model on the test data
def test_model(genome, test_data, config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    net = neat.nn.RecurrentNetwork.create(genome, config)  # Create the neural network from the best genome
    env = TradingEnvironment(test_data)  # Initialize the environment with test data
    env.reset()  # Reset the environment

    actions = []
    while True:
        state = env.data.iloc[env.current_step, :11].values  # Get the current state from the environment
        action = np.argmax(net.activate(state)) - 1  # Choose action from the neural network
        actions.append(action)
        balance, done, win, total_profit = env.step(action)  # Step the environment with the chosen action
        if done:  # Break the loop if the episode is done
            break
    return actions


def load_best_genome(path_to_best_genome):
    with open(path_to_best_genome, 'rb') as f:
        return pickle.load(f)

def compute_trade_details(entry_price, sl_price, amount_usdt, position_type):
    loss_pct = 0.03
    gain_pct = 0.15
    if position_type == "short":
        distance = (sl_price - entry_price) / entry_price
        distance2 = loss_pct / (distance)
        leverage = abs(distance2)
        tp_price = ((-gain_pct * entry_price) / (leverage)) + entry_price
        position_size = amount_usdt * leverage
    elif position_type == "long":
        distance = (sl_price - entry_price) / entry_price
        distance2 = -loss_pct / (distance)
        leverage = abs(distance2)
        tp_price = ((gain_pct * entry_price) / (leverage)) + entry_price
        position_size = amount_usdt * leverage

    return position_size, round(tp_price,2), round(leverage,2)

#Set Leverage Function
def set_leverage(symbol,leverage, marginmode):
    result = accountAPI.set_leverage(
        instId=symbol,
        lever=str(leverage),
        mgnMode=marginmode
    )
    return result

# Function to get the latest price of SOL-USDT-SWAP
def get_latest_price(symbol):
    try:
        # Fetch the ticker information for the specified symbol
        ticker = okx.fetch_ticker(symbol)

        # Extract the last price from the ticker information
        last_price = ticker['last']
        return last_price
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def place_futures_order(symbol, side, size, trigger_price):
    try:
        # Place an algo order with trigger price
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode="isolated",  # Isolated margin mode
            side=side,  # "buy" for long, "sell" for short
            ordType="trigger",  # Trigger order type
            sz=str(size),  # Number of contracts
            triggerPx=str(trigger_price),  # Trigger price for the order to be activated
            orderPx=-1,
            triggerPxType="mark",
        )
        return result
    except Exception as e:
        print(f"Error placing buy order: {e}")
        return None

def place_stop_loss_in_buy(symbol,sl_price,order_size,marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="sell",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            slTriggerPx=str(sl_price),  # Stop-loss trigger price
            slOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            slTriggerPxType="mark"  # Use the mark price as reference
        )

        # Print result (order details)
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def place_stop_loss_in_sell(symbol, sl_price, order_size, marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="buy",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            slTriggerPx=str(sl_price),  # Stop-loss trigger price
            slOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            slTriggerPxType="mark"  # Use the mark price as reference
        )
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def place_tp_in_buy(symbol, tp_price, order_size, marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="sell",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            tpTriggerPx=str(tp_price),  # Stop-loss trigger price
            tpOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            tpTriggerPxType="mark"  # Use the mark price as reference
        )

        # Print result (order details)
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def place_tp_in_sell(symbol, tp_price, order_size, marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="buy",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            tpTriggerPx=str(tp_price),  # Stop-loss trigger price
            tpOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            tpTriggerPxType="mark"  # Use the mark price as reference
        )

        # Print result (order details)
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def sync_to_5_minute():
    now = datetime.now()
    # Calculate the next 5-minute mark
    next_five_minute = (now + timedelta(minutes=5 - (now.minute % 5))).replace(second=0, microsecond=0)
    # Calculate the seconds until the next 5-minute mark
    seconds_left = (next_five_minute - now).total_seconds()
    # Sleep until the next 5-minute mark
    if seconds_left > 0:
        time.sleep(seconds_left)


if __name__ == "__main__":
    # Provide path to your NEAT config file
    # Create configuration file
    create_neat_config()
    config_path = "neat_config6.txt"

    # Load the best genome for testing
    path_to_best_genome = 'best_genome2.pkl'  # Path to the saved genome file
    best_genome = load_best_genome(path_to_best_genome)

    symbol_2 = input('Enter your TICKER in OKX for trading: (eg. SOL-USDT-SWAP, BTC-USDT-SWAP)')
    #Go Live Trading
    while True:
        try:
            # Initialize tvDatafeed
            tv = TvDatafeed()

            # Specify parameters
            symbol_used = 'SOLUSDT.P'
            platform = 'OKX'
            n_bars = 5000

            # Fetch historical data for symbol_used
            df_symbolused = fetch_data_with_retry(tv, symbol_used, platform, Interval.in_5_minute, n_bars)

            if df_symbolused is None:
                raise ValueError("Failed to fetch data for symbol_used.")

            # Rename columns for consistency
            df_symbolused.columns = ['symbol', 'open_symbolused', 'high_symbolused', 'low_symbolused', 'close_symbolused',
                                     'volume_symbolused']

            # Symbols for additional data
            btc_symbol = 'SOL'
            total_symbol = 'TOTAL'
            platform_cryptocap = 'CRYPTOCAP'

            # Fetch OHLC data for BTC and TOTAL
            df_btc = fetch_data_with_retry(tv, btc_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)
            df_total = fetch_data_with_retry(tv, total_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)

            # Check if the data is fetched successfully
            if df_btc is None or df_total is None:
                raise ValueError("Failed to fetch data for BTC or TOTAL.")

            # Rename columns for consistency
            df_btc.columns = ['symbol', 'open_btc', 'high_btc', 'low_btc', 'close_btc', 'volume_btc']
            df_total.columns = ['symbol', 'open_total', 'high_total', 'low_total', 'close_total', 'volume_total']

            # Merge the datasets on index (align the timestamps)
            df_combined = pd.concat([df_symbolused, df_btc, df_total], axis=1)
            df_combined.dropna(inplace=True)  # Drop NaN values due to potential mismatched timestamps

            # Calculate Stochastic Slow and EMAs for the symbolused
            data = df_combined
            data['ema5'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=5).ema_indicator()
            data['ema8'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=8).ema_indicator()
            data['ema100'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=100).ema_indicator()

            # Calculate slopes for symbolused, btc, and total
            data['symbolused_slope'] = calculate_slope(data['close_symbolused'], window=7)
            data['btc_slope'] = calculate_slope(data['close_btc'], window=7)
            data['total_slope'] = calculate_slope(data['close_total'], window=7)
            data.dropna(inplace=True)

            data_inputs = pd.DataFrame({
                'close_symbolused': data['close_symbolused'],
                'high_symbolused': data['high_symbolused'],
                'low_symbolused': data['low_symbolused'],
                'open_symbolused': data['open_symbolused'],
                'symbolused_slope': data['symbolused_slope'],
                'btc_slope': data['btc_slope'],
                'total_slope': data['total_slope'],
                'ema5': data['ema5'],
                'ema8': data['ema8'],
                'ema100': data['ema100'],
                'close_btc': data['close_btc'],
                'total_btc': data['close_total']
            })

            ema5 = data_inputs['ema5']
            ema8 = data_inputs['ema8']
            ema100 = data_inputs['ema100']
            close_symbolused = data_inputs['close_symbolused']
            open_symbolused = data_inputs['open_symbolused']
            high_symbolused = data_inputs['high_symbolused']
            low_symbolused = data_inputs['low_symbolused']

            actions = test_model(best_genome, data_inputs, config_path)

            import ccxt
            import okx.Trade as Trade
            import okx.Account as Account

            # API credentials
            apikey = "e5f90b15-e3f1-469d-9477-11fdfcf04fdb"
            secretkey = "F79E3CE531674965F95CAEDEA3C81C69"
            passphrase = "Easy09159562534*"

            marginmode = "isolated"

            # Initialize OKX exchange with your API credentials
            okx = ccxt.okx({
                'apiKey': apikey,
                'secret': secretkey,
                'password': passphrase,
                'enableRateLimit': True,  # Ensures you don't exceed rate limits
            })

            flag = "0"  # Production trading: 0, Demo trading: 1

            # Initialize TradeAPI and AccountAPI
            tradeAPI = Trade.TradeAPI(apikey, secretkey, passphrase, False, flag)
            accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)

            # Fetch your overall balance
            balance = okx.fetch_balance()

            # Access USDT balance
            usdt_balance = balance['total'].get('USDT', 0)


            console = Console()

            # Execute the action (you can integrate with a broker API to execute buy/sell)
            if ema5[-3]>ema100[-3] and ema8[-3]<ema100[-3] and ema5[-2]>ema100[-2] and ema8[-2]>ema100[-2] and open_symbolused[-2]<close_symbolused[-2]:  # Buy action
                console.print('[yellow]Bullish Crossover Detected[/yellow]')

                # Bullish crossover: Buy when short EMA > long EMA
                if actions[-2] == 1:
                    console.print('[yellow]Predictions Confirmed[/yellow]')
                    entry_price = close_symbolused[-2]
                    sl_price = ema100[-2]
                    position_size_usdt, take_profit_price, leverage = compute_trade_details(entry_price, sl_price,usdt_balance, "long")
                    console.print(f'[yellow]Action: Buy (Bullish EMA Crossover). SL: {sl_price}. Entry: {entry_price}. TP: {take_profit_price}, Leverage: {leverage}, Position Size: {position_size_usdt}[/yellow]')
                    size = position_size_usdt / get_latest_price(symbol_2)
                    set_leverage(symbol_2, leverage, marginmode)
                    # Place a buy order with your broker API
                    if leverage < 50:
                        order = place_futures_order(symbol_2, "buy", round(size, 2), entry_price)
                        print(order)

                        # Extract algoId from the order response
                        order_algo_id = order['data'][0]['algoId']

                        # Wait until the order is triggered
                        while True:
                            try:
                                positions = okx.fetch_positions()
                                current_price = get_latest_price(symbol_2)

                                # Check if the order is filled
                                if len(positions) > 0:  # Adjusted to check for any filled positions
                                    for position in positions:
                                        position_size = position['contracts']
                                        positions = okx.fetch_positions()

                                        # Place stop-loss and take-profit orders
                                        stop_loss_order = place_stop_loss_in_buy(symbol_2, sl_price, position_size, marginmode)
                                        stop_loss_algo_id = stop_loss_order['data'][0]['algoId']
                                        print(f'Stop_Loss Order: {stop_loss_order}')

                                        tp_order = place_tp_in_buy(symbol_2, take_profit_price, position_size, marginmode)
                                        tp_algo_id = tp_order['data'][0]['algoId']
                                        print(f'TP Order: {tp_order}')

                                        # Monitor stop-loss and take-profit orders
                                        while True:
                                            try:
                                                current_price = get_latest_price(symbol_2)
                                                positions = okx.fetch_positions()
                                                # Check if stop-loss or take-profit has been triggered
                                                if current_price <= sl_price and len(positions) == 0:
                                                    algo_orders = [
                                                        {"instId": symbol_2, "algoId": tp_algo_id}
                                                    ]
                                                    # Cancel unfilled algo orders
                                                    result = tradeAPI.cancel_algo_order(algo_orders)
                                                    break
                                                elif current_price >= take_profit_price and len(positions) == 0:
                                                    algo_orders = [
                                                        {"instId": symbol_2, "algoId": stop_loss_algo_id}
                                                    ]
                                                    # Cancel unfilled algo orders
                                                    result = tradeAPI.cancel_algo_order(algo_orders)
                                                    break
                                            except Exception as e:
                                                print(e)
                                    # Exit the position management loop after handling the position
                                    break
                                elif len(positions) == 0 and current_price <= sl_price or current_price >= take_profit_price:
                                    algo_orders = [{"instId": symbol_2, "algoId": order_algo_id}]
                                    result = tradeAPI.cancel_algo_order(algo_orders)
                                    break
                            except Exception as e:
                                print(e)
                    else:
                        console.print('[yellow]Leverage is too high, order Cancelled.[/yellow]')

            elif ema5[-3]<ema100[-3] and ema8[-3]>ema100[-3] and ema5[-2]<ema100[-2] and ema8[-2]<ema100[-2] and open_symbolused[-2]>close_symbolused[-2]:  # Sell action
                console.print('[yellow]Bearish Crossover Detected[/yellow]')

                # Bearish crossover: Sell when short EMA < long EMA
                if actions[-2] == -1:
                    console.print('[yellow]Predictions Confirmed[/yellow]')
                    entry_price = close_symbolused[-2]
                    sl_price = ema100[-2]
                    position_size_usdt, take_profit_price, leverage = compute_trade_details(entry_price, sl_price,usdt_balance, "short")
                    size = position_size_usdt / get_latest_price(symbol_2)
                    set_leverage(symbol_2, leverage, marginmode)
                    console.print(f'[yellow]Action: Sell (Bearish EMA Crossover). SL: {sl_price}. Entry: {entry_price}. TP: {take_profit_price}, Leverage: {leverage}, Position Size: {position_size_usdt}[/yellow]')

                    if leverage < 50:
                        # Place a sell order with your broker API
                        order = place_futures_order(symbol_2, "sell", round(size, 2), entry_price)
                        print(order)

                        # Extract algoId from the order response
                        order_algo_id = order['data'][0]['algoId']

                        # Wait until the order is triggered
                        while True:
                            try:
                                positions = okx.fetch_positions()
                                current_price = get_latest_price(symbol_2)
                                # Check if the order is filled
                                if len(positions) > 0:  # Adjusted to check for any filled positions
                                    for position in positions:
                                        position_size = position['contracts']
                                        current_price = get_latest_price(symbol_2)

                                        # Place stop-loss and take-profit orders
                                        stop_loss_order = place_stop_loss_in_sell(symbol_2, sl_price, position_size, marginmode)
                                        stop_loss_algo_id = stop_loss_order['data'][0]['algoId']
                                        print(f'Stop_Loss Order: {stop_loss_order}')

                                        tp_order = place_tp_in_sell(symbol_2, take_profit_price, position_size, marginmode)
                                        tp_algo_id = tp_order['data'][0]['algoId']
                                        print(f'TP Order: {tp_order}')

                                        # Monitor stop-loss and take-profit orders
                                        while True:
                                            try:
                                                current_price = get_latest_price(symbol_2)
                                                positions = okx.fetch_positions()

                                                # Check if stop-loss or take-profit has been triggered
                                                if current_price >= sl_price and len(positions) == 0:
                                                    algo_orders = [
                                                        {"instId": symbol_2, "algoId": tp_algo_id}
                                                    ]
                                                    # Cancel unfilled algo orders
                                                    result = tradeAPI.cancel_algo_order(algo_orders)
                                                    break
                                                elif current_price <= take_profit_price and len(positions) == 0:
                                                    algo_orders = [
                                                        {"instId": symbol_2, "algoId": stop_loss_algo_id}
                                                    ]
                                                    # Cancel unfilled algo orders
                                                    result = tradeAPI.cancel_algo_order(algo_orders)
                                                    break
                                            except Exception as e:
                                                print(e)

                                    # Exit the position management loop after handling the position
                                    break

                                elif len(positions) == 0 and current_price >= sl_price or current_price <= take_profit_price:
                                    algo_orders = [{"instId": symbol_2, "algoId": order_algo_id}]
                                    result = tradeAPI.cancel_algo_order(algo_orders)
                                    break
                            except Exception as e:
                                print(e)
                    else:
                        console.print('[yellow]Leverage is too high, order cancelled[/yellow]')
            else:
                prev_price = close_symbolused[-2]
                prev_open = open_symbolused[-2]
                prev_high = high_symbolused[-2]
                prev_low = low_symbolused[-2]
                console.print(f'[green]Action: Hold. High: {prev_high}. Low: {prev_low}. Open: {prev_open}. Close: {prev_price}. EMA5: {ema5[-2]}. EMA8: {ema8[-2]}. EMA100: {ema100[-2]}[green/]')
            # Wait for the next interval (e.g., 5 minute)
            console.print('[yellow]Waiting for the next candle...[/yellow]')
        except Exception as e:
            print(e)
        sync_to_5_minute()
