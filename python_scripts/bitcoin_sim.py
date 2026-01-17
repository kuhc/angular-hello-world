import numpy as np
import pandas as pd
import sys

def simulate_bitcoin_price(days=60, s0=60000, mu=0.0005, sigma=0.04):
    """
    Simulates Bitcoin price using Geometric Brownian Motion.

    Args:
        days (int): Number of days to simulate.
        s0 (float): Initial price.
        mu (float): Daily drift (expected return).
        sigma (float): Daily volatility.

    Returns:
        pd.DataFrame: DataFrame containing Date and Price.
    """
    np.random.seed(1) # For reproducibility (Seed 1 produces a crossover in this short timeframe)
    dt = 1
    prices = [s0]

    for _ in range(days - 1):
        # random shock
        epsilon = np.random.normal(0, 1)
        # Calculate log return
        # S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * epsilon * sqrt(dt))
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * epsilon * np.sqrt(dt)
        new_price = prices[-1] * np.exp(log_ret)
        prices.append(new_price)

    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    return pd.DataFrame({'Date': dates, 'Price': prices})

def calculate_mas(df):
    """Calculates 7-day and 30-day Moving Averages."""
    df['MA7'] = df['Price'].rolling(window=7).mean()
    df['MA30'] = df['Price'].rolling(window=30).mean()
    return df

def run_strategy(df, initial_cash=100000.0):
    """
    Runs the Golden Cross strategy.

    Buy when MA7 crosses above MA30.
    Sell when MA7 crosses below MA30.
    """
    cash = initial_cash
    btc = 0.0
    position = 0 # 0: USD, 1: BTC

    ledger = []
    portfolio_history = []

    print("\nDaily Ledger of Trades:")
    print(f"{'Date':<12} | {'Action':<6} | {'Price ($)':<10} | {'Amount':<15} | {'Portfolio Value ($)':<20}")
    print("-" * 80)

    for i in range(len(df)):
        date = df.loc[i, 'Date'].strftime('%Y-%m-%d')
        price = df.loc[i, 'Price']
        ma7 = df.loc[i, 'MA7']
        ma30 = df.loc[i, 'MA30']

        # Determine Portfolio Value
        current_val = cash + (btc * price)
        portfolio_history.append(current_val)

        # Trading Logic
        # Need previous day to check for crossover
        if i > 0:
            prev_ma7 = df.loc[i-1, 'MA7']
            prev_ma30 = df.loc[i-1, 'MA30']

            # Ensure we have valid MAs
            if pd.notna(ma7) and pd.notna(ma30) and pd.notna(prev_ma7) and pd.notna(prev_ma30):

                # GOLDEN CROSS (Buy Signal)
                if prev_ma7 <= prev_ma30 and ma7 > ma30:
                    if position == 0:
                        btc = cash / price
                        amount_traded = btc
                        cash = 0.0
                        position = 1
                        print(f"{date:<12} | {'BUY':<6} | {price:<10.2f} | {amount_traded:<15.6f} | {current_val:<20.2f}")
                        ledger.append({'Date': date, 'Action': 'BUY', 'Price': price, 'Amount': amount_traded})

                # DEATH CROSS (Sell Signal)
                elif prev_ma7 >= prev_ma30 and ma7 < ma30:
                    if position == 1:
                        amount_traded = btc
                        cash = btc * price
                        btc = 0.0
                        position = 0
                        print(f"{date:<12} | {'SELL':<6} | {price:<10.2f} | {amount_traded:<15.6f} | {current_val:<20.2f}")
                        ledger.append({'Date': date, 'Action': 'SELL', 'Price': price, 'Amount': amount_traded})

    return portfolio_history, ledger

def main():
    # 1. Simulate Data
    print("Simulating 60 days of Bitcoin prices...")
    df = simulate_bitcoin_price()

    # 2. Calculate MAs
    df = calculate_mas(df)

    # 3. Run Strategy
    portfolio_values, trade_ledger = run_strategy(df)

    # 4. Final Performance
    final_val = portfolio_values[-1]
    initial_val = 100000.0
    pnl = final_val - initial_val
    return_pct = (pnl / initial_val) * 100

    print("-" * 80)
    print(f"Final Portfolio Value: ${final_val:,.2f}")
    print(f"Total PnL:             ${pnl:,.2f}")
    print(f"Return:                {return_pct:.2f}%")
    print("-" * 80)

if __name__ == "__main__":
    main()
