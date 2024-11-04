# Import necessary libraries
import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import ta
from scipy.stats import linregress
import neat
import time
import pickle

# Initialize tvDatafeed
tv = TvDatafeed()


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
sol_symbol = 'SOL'
btc_symbol = 'BTC'
total_symbol = 'TOTAL'
platform_cryptocap = 'CRYPTOCAP'


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


def calculate_rolling_std_high_low(high_series, low_series, window=7):
    """
    Calculate the rolling standard deviation using the average of high and low prices over a specified window.

    Parameters:
        high_series (pd.Series): The high prices series.
        low_series (pd.Series): The low prices series.
        window (int): The rolling window size.

    Returns:
        pd.Series: A series containing the rolling standard deviation of the average of high and low prices.
    """
    # Calculate the average of high and low prices
    avg_high_low = (high_series + low_series) / 2

    # Initialize a list with NaN for the initial values where the rolling window can't be applied
    rolling_std = [np.nan] * (window - 1)

    # Calculate the rolling standard deviation
    for i in range(window - 1, len(avg_high_low)):
        window_data = avg_high_low[i - window + 1:i + 1]
        std_dev = np.std(window_data)
        rolling_std.append(std_dev)

    return pd.Series(rolling_std, index=high_series.index)


def calculate_rolling_correlation(series1, series2, window=7):
    """
    Calculate the rolling correlation between two price series over a specified window.

    Parameters:
        series1 (pd.Series): The first price series (e.g., BTC close prices).
        series2 (pd.Series): The second price series (e.g., ETH close prices).
        window (int): The rolling window size (default is 7).

    Returns:
        pd.Series: A series containing the rolling correlation values.
    """
    # Calculate the rolling correlation
    rolling_corr = series1.rolling(window).corr(series2)

    return rolling_corr


# Fetch OHLC data for SOL ,BTC and TOTAL
df_sol = fetch_data_with_retry(tv, sol_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)
df_total = fetch_data_with_retry(tv, total_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)
df_btc = fetch_data_with_retry(tv, btc_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)

# Check if the data is fetched successfully
if df_sol is None or df_total is None or df_btc is None:
    raise ValueError("Failed to fetch data for BTC or TOTAL.")

# Rename columns for consistency
df_sol.columns = ['symbol', 'open_sol', 'high_sol', 'low_sol', 'close_sol', 'volume_sol']
df_total.columns = ['symbol', 'open_total', 'high_total', 'low_total', 'close_total', 'volume_total']
df_btc.columns = ['symbol', 'open_btc', 'high_btc', 'low_btc', 'close_btc', 'volume_btc']


# Merge the datasets on index (align the timestamps)
df_combined = pd.concat([df_symbolused, df_sol, df_total, df_btc], axis=1)
df_combined.dropna(inplace=True)  # Drop NaN values due to potential mismatched timestamps


# Calculate Stochastic Slow and EMAs for the symbolused
data = df_combined
data['ema5'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=5).ema_indicator()
data['ema8'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=8).ema_indicator()
data['ema100'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=50).ema_indicator()

# Calculate slopes for symbolused, btc, and total
data['symbolused_slope'] = calculate_slope(data['close_symbolused'], window=7)
data['sol_slope'] = calculate_slope(data['close_sol'], window=7)
data['total_slope'] = calculate_slope(data['close_total'], window=7)
data['btc_slope'] = calculate_slope(data['close_btc'], window=7)

#Compute for Price ratio
data['price_ratio_symbolused'] = (data['close_symbolused']-data['open_symbolused'])/data['open_symbolused']
data['price_ratio_sol'] = (data['close_sol']-data['open_sol'])/data['open_sol']
data['price_ratio_total'] = (data['close_total']-data['open_total'])/data['open_total']
data['price_ratio_btc'] = (data['close_btc']-data['open_btc'])/data['open_btc']


#Compute for RSI 14 for symbolused
data['rsi14_symbolused'] = ta.momentum.RSIIndicator(close=data['close_symbolused'], window=14).rsi()

#Compute for standard deviation
data['std_symbolused'] = calculate_rolling_std_high_low(data['high_symbolused'],data['low_symbolused'])
data['std_sol'] = calculate_rolling_std_high_low(data['high_sol'],data['low_sol'])
data['std_total'] = calculate_rolling_std_high_low(data['high_total'],data['low_total'])
data['std_btc'] = calculate_rolling_std_high_low(data['high_btc'],data['low_btc'])

#Compute for correlation between sol and btc
data['correlation_sol_btc'] = calculate_rolling_correlation(data['close_sol'], data['close_btc'])

data.dropna(inplace=True)

data_inputs = pd.DataFrame({
    'close_symbolused': data['close_symbolused'],
    'high_symbolused': data['high_symbolused'],
    'low_symbolused': data['low_symbolused'],
    'open_symbolused': data['open_symbolused'],
    'symbolused_slope': data['symbolused_slope'],
    'sol_slope': data['sol_slope'],
    'total_slope': data['total_slope'],
    'btc_slope': data['btc_slope'],
    'std_symbolused': data['std_symbolused'],
    'std_total': data['std_total'],
    'std_btc': data['std_btc'],
    'std_sol': data['std_sol'],
    'price_ratio_symbolused': data['price_ratio_symbolused'],
    'price_ratio_total': data['price_ratio_total'],
    'price_ratio_sol': data['price_ratio_sol'],
    'price_ratio_btc': data['price_ratio_btc'],
    'ema5': data['ema5'],
    'ema8': data['ema8'],
    'ema100': data['ema100'],
    'close_sol': data['close_sol'],
    'close_total': data['close_total'],
    'close_btc': data['close_btc'],
    'rsi14_symbolused': data['rsi14_symbolused'],
    'correlation_sol_btc':data['correlation_sol_btc']
})

print(f'length of data: {len(data_inputs)}')
# Split data into two halves for training and testing
first_half = data_inputs[:len(data_inputs) // 2]
second_half = data_inputs[len(data_inputs) // 2:]


# Define a trading environment with a feedback loop
class TradingEnvironment:
    def __init__(self, data, starting_balance=20):
        self.data = data
        self.balance = starting_balance
        self.current_step = 0
        self.winloss = []
        self.action_list = []
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
        self.action_list.append(action)

        # Look ahead for the next 100 steps to store highs and lows
        for step_ahead in range(1, 400):
            if self.current_step + step_ahead < len(self.close_price):
                future_low = self.low_price[self.current_step + step_ahead]
                future_high = self.high_price[self.current_step + step_ahead]
                low_and_high.append((future_low, future_high))

        ema5_current, ema8_current, ema100_current = self.ema5[self.current_step], self.ema8[self.current_step], self.ema100[self.current_step]
        ema5_prev, ema8_prev, ema100_prev = self.ema5[self.current_step - 1], self.ema8[self.current_step - 1], self.ema100[self.current_step - 1]

        step = 1
        if action == 1 and ema5_prev > ema100_prev and ema8_prev < ema100_prev and ema8_current> ema100_current and ema5_current>ema100_current and price > self.open_price[self.current_step]:
            loss_pct = 0.03
            gain_pct = 0.15
            stop_loss = ema100_current
            tp_limit = ((gain_pct * price) / abs(-loss_pct / ((stop_loss - price) / price))) + price

            # Evaluate if TP or SL is hit in the lookahead prices
            a = []
            for i in low_and_high:
                low, high = i
                a.append(1)
                self.action_list.append(0)
                if high >= tp_limit:  # Take Profit hit1
                    step = len(a) + 1
                    self.balance += self.balance * gain_pct
                    self.winloss.append(1)
                    self.profits.append(sum(self.profits)*gain_pct)
                    break
                elif low <= stop_loss:  # Stop Loss hit
                    step = len(a) + 1
                    self.balance -= self.balance * loss_pct
                    self.profits.append(-sum(self.profits) * loss_pct)
                    self.winloss.append(-1)
                    break
        elif action == -1 and ema5_prev < ema100_prev and ema8_prev > ema100_prev and ema8_current< ema100_current and ema5_current<ema100_current and price < self.open_price[
            self.current_step]:
            loss_pct = 0.03
            gain_pct = 0.15
            stop_loss = ema100_current
            tp_limit = ((-gain_pct * price) / abs(loss_pct / ((stop_loss - price) / price))) + price
            a = []
            for i in low_and_high:
                low, high = i
                a.append(1)
                self.action_list.append(0)
                if low <= tp_limit:  # Take Profit hit for sell
                    step = len(a) + 1
                    self.balance += self.balance * gain_pct
                    self.profits.append(sum(self.profits) * gain_pct)
                    self.winloss.append(1)
                    break
                elif high >= stop_loss:  # Stop Loss hit for sell
                    step = len(a) + 1
                    self.balance -= self.balance * loss_pct
                    self.profits.append(-sum(self.profits) * loss_pct)
                    self.winloss.append(-1)
                    break

        self.current_step += step
        self.equity_history.append(self.balance)
        done = self.current_step >= len(self.data) - 1
        return self.balance, done, self.winloss, sum(self.profits), self.action_list


# Define fitness function for NEAT
def evaluate_genome(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0  # Initialize fitness to ensure it’s not None

        try:
            # Create the neural network for the genome
            net = neat.nn.RecurrentNetwork.create(genome, config)
            env = TradingEnvironment(data_inputs)
            env.reset()

            total_profit = 0  # Variable to accumulate the genome's profit

            while True:
                state = env.data.iloc[env.current_step, :24].values
                action = np.argmax(net.activate(state)) - 1  # Map to -1, 0, 1
                balance, done, winloss, total_profits, action_list = env.step(action)
                if done:
                    break

            print(len(data))
            print(len(action_list))
            profit = balance
            set = winloss
            total_profit = total_profits
            win = set.count(1)
            loss = set.count(-1)
            PNL = (total_profit-20)*100/20
            if len(set)==0:
                winrate = 0
            else:
                winrate = set.count(1)/len(set)
            print(f'Trader: {genome_id}, PNL%: {round(PNL,2)}%, Winrate: {round(winrate,2)*100}%, Wins: {win}, Loss: {loss}')
            genome.fitness = max(total_profit, 0)  # Set fitness, ensuring no negative values

        except Exception as e:
            print(f"Error evaluating genome {genome_id}: {e}")
            genome.fitness = 0  # Set to zero to avoid NoneType issues


# NEAT setup function
def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create a population
    p = neat.Population(config)

    # Add reporters to monitor progress
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.Checkpointer(5))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for a certain number of generations
    winner = p.run(evaluate_genome, 100)
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the best genome
    with open('best_genome2.pkl', 'wb') as f:  # Save the genome to a file
        pickle.dump(winner, f)


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
    num_inputs = 24
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
    with open('neat_config6.txt', 'w') as f:
        f.write(config_content)


# Function to test the trained NEAT model on the test data
def test_model(genome, test_data, config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    net = neat.nn.FeedForwardNetwork.create(genome, config)  # Create the neural network from the best genome
    env = TradingEnvironment(test_data)  # Initialize the environment with test data
    env.reset()  # Reset the environment

    total_profit = 0  # Variable to accumulate profit during testing
    while True:
        state = env.data.iloc[env.current_step, :24].values  # Get the current state from the environment
        action = np.argmax(net.activate(state)) - 1  # Choose action from the neural network
        balance, done, winloss, total_profit = env.step(action)  # Step the environment with the chosen action
        if done:  # Break the loop if the episode is done
            break

    total_profit = balance - 10  # Calculate profit (final balance - initial balance)
    return total_profit


def load_best_genome(path_to_best_genome):
    with open(path_to_best_genome, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Provide path to your NEAT config file
    # Create configuration file
    create_neat_config()
    config_path = "neat_config6.txt"
    run_neat(config_path)

