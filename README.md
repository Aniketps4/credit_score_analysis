# Wallet Credit Scoring Analysis Project

## Project Description
This project analyzes the financial behavior of wallets by processing transaction data from `user-wallet-transactions.json` to compute credit scores. Conducted at 07:05 PM IST on Wednesday, July 16, 2025, the analysis generates scores based on features like deposits, borrows, and transaction activity, saving results to `wallet_scores.json` and visualizing the distribution in `score_histogram.png`.

## Method Chosen
The methodology integrates data processing and machine learning-inspired scoring:
- **Data Loading**: Parse JSON transaction data and extract key fields (e.g., amount, assetSymbol, timestamp) into a pandas DataFrame.
- **Feature Engineering**: Derive features such as total deposits, borrows, repayment ratio, liquidation count, average transaction size, activity days, token diversity, and transaction count from grouped wallet data.
- **Normalization**: Scale features to a [0, 1] range using max normalization, adjusting liquidation counts inversely.
- **Scoring**: Compute credit scores using a weighted sum of normalized features, applied through a sigmoid function to map scores to a 0-1000 range.
- **Visualization**: Generate a histogram of score distribution using matplotlib.

## Complete Architecture
The architecture is a standalone Python script:
- **Core Library**: Uses pandas for data manipulation, numpy for numerical operations, and matplotlib for visualization.
- **Data Processing**: In-memory computation with pandas DataFrames, handling JSON parsing and CSV output.
- **Output**: Produces `wallet_scores.json` for scores, `features.csv` for intermediate features, and `score_distribution.json` for range counts, with a saved `score_histogram.png`.
- **Dependencies**: Requires pandas, numpy, and matplotlib, installable via pip (e.g., `pip install pandas numpy matplotlib`).

## Processing Flow
1. **Data Loading**: Reads `user-wallet-transactions.json`, extracts actionData fields (amount, assetSymbol, assetPriceUSD), computes amountUSD, and returns a DataFrame with userWallet, action, timestamp, amountUSD, and assetSymbol.
2. **Data Cleaning**: Groups data by userWallet, handles edge cases (e.g., zero borrows for repayment ratio), and ensures valid timestamps.
3. **Analysis**: 
   - Engineers features from grouped data.
   - Normalizes features to [0, 1].
   - Computes scores using predefined weights and a sigmoid transformation.
4. **Visualization**: Creates a histogram of credit scores binned into 100-unit ranges (0-100, 100-200, etc.) and saves it as `score_histogram.png`.
5. **Output**: Saves scores to `wallet_scores.json`, features to `features.csv`, distribution counts to `score_distribution.json`, and prints statistical summaries (e.g., mean, correlations).

## Data Files
- **user-wallet-transactions.json**: Input JSON file with transaction data including actionData (amount, assetSymbol, assetPriceUSD) and timestamp.
- **wallet_scores.json**: Output JSON file with userWallet and computed credit scores (0-1000).
- **features.csv**: CSV file with intermediate feature data for each wallet.
- **score_distribution.json**: JSON file with count of wallets per score range.
- **score_histogram.png**: Saved image of the credit score distribution histogram.

## Usage
1. Install dependencies: `pip install pandas numpy matplotlib`.
2. Place `user-wallet-transactions.json` in the working directory.
3. Run the script: `python score_wallet_up.py`.
4. Review output files (`wallet_scores.json`, `features.csv`, `score_distribution.json`) and `score_histogram.png`.

## Limitations
- Assumes `user-wallet-transactions.json` follows the expected structure; errors may occur with missing fields.
- Score computation relies on fixed weights and a sigmoid function, which may need tuning for accuracy.
- Visualization is static; interactive options are not included.

## Contributing
Provide additional transaction data or suggest weight adjustments to improve scoring accuracy.

## License
For educational purposes only, with no commercial intent.
