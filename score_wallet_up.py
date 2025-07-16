'''import pandas as pd
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Load JSON data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['amount'] = df['actionData'].apply(lambda x: float(x['amount']))
    df['assetSymbol'] = df['actionData'].apply(lambda x: x['assetSymbol'])
    df['assetPriceUSD'] = df['actionData'].apply(lambda x: float(x['assetPriceUSD']))
    df['amountUSD'] = df['amount'] * df['assetPriceUSD'] / (10 ** df['assetSymbol'].map({'USDC': 6, 'WMATIC': 18, 'DAI': 18, 'WETH': 18}).fillna(18))
    return df[['userWallet', 'action', 'timestamp', 'amountUSD', 'assetSymbol']]

# Feature engineering
def engineer_features(df):
    grouped = df.groupby('userWallet')
    features = {
        'userWallet': [],
        'total_deposits_usd': [],
        'total_borrows_usd': [],
        'repayment_ratio': [],
        'liquidation_count': [],
        'avg_transaction_size_usd': [],
        'activity_days': [],
        'token_diversity': [],
        'transaction_count': []
    }
    
    for wallet, group in grouped:
        features['userWallet'].append(wallet)
        deposits = group[group['action'] == 'deposit']['amountUSD'].sum()
        redeems = group[group['action'] == 'redeemunderlying']['amountUSD'].sum()
        features['total_deposits_usd'].append(max(deposits - redeems, 0))
        borrows = group[group['action'] == 'borrow']['amountUSD'].sum()
        features['total_borrows_usd'].append(borrows)
        repays = group[group['action'] == 'repay']['amountUSD'].sum()
        repayment_ratio = repays / borrows if borrows > 0 else 0.5  # Neutral default
        features['repayment_ratio'].append(min(repayment_ratio, 1.0))
        liquidations = len(group[group['action'] == 'liquidationcall'])
        features['liquidation_count'].append(liquidations)
        avg_size = group['amountUSD'].mean()
        features['avg_transaction_size_usd'].append(avg_size if not np.isnan(avg_size) else 0)
        timestamps = pd.to_datetime(group['timestamp'], unit='s')
        activity_span = (timestamps.max() - timestamps.min()).days
        features['activity_days'].append(activity_span if activity_span > 0 else 1)
        features['token_diversity'].append(len(group['assetSymbol'].unique()))
        features['transaction_count'].append(len(group))
    
    return pd.DataFrame(features)

# Normalize features to [0, 1]
def normalize_features(df):
    for col in ['total_deposits_usd', 'total_borrows_usd', 'avg_transaction_size_usd', 'activity_days', 'token_diversity', 'transaction_count']:
        max_val = df[col].max()
        if max_val > 0:
            df[col] = df[col] / max_val
    max_liquidations = df['liquidation_count'].max()
    if max_liquidations > 0:
        df['liquidation_count'] = 1 - (df['liquidation_count'] / max_liquidations)
    return df

# Compute credit score
def compute_scores(features_df):
    weights = {
        'total_deposits_usd': 0.25,
        'repayment_ratio': 0.20,
        'liquidation_count': 0.35,
        'avg_transaction_size_usd': 0.10,
        'activity_days': 0.07,
        'token_diversity': 0.03,
        'transaction_count': 0.03
    }
    
    scores = (
        features_df['total_deposits_usd'] * weights['total_deposits_usd'] +
        features_df['repayment_ratio'] * weights['repayment_ratio'] +
        features_df['liquidation_count'] * weights['liquidation_count'] +
        features_df['avg_transaction_size_usd'] * weights['avg_transaction_size_usd'] +
        features_df['activity_days'] * weights['activity_days'] +
        features_df['token_diversity'] * weights['token_diversity'] +
        features_df['transaction_count'] * weights['transaction_count']
    )
    scores = 1000 * (1 / (1 + np.exp(-5 * (scores - 0.5))))
    
    features_df['credit_score'] = scores.clip(0, 1000).round().astype(int)
    return features_df[['userWallet', 'credit_score']]

# Generate score distribution
def generate_distribution(scores_df):
    bins = range(0, 1001, 100)
    labels = [f"{i}-{i+100}" for i in range(0, 1000, 100)]
    distribution = pd.cut(scores_df['credit_score'], bins=bins, labels=labels, include_lowest=True)
    dist_counts = distribution.value_counts().sort_index().to_dict()
    with open('score_distribution.json', 'w') as f:
        json.dump(dist_counts, f, indent=4)
    return dist_counts

# Save scores to JSON
def save_scores(scores_df, output_path):
    scores_dict = scores_df.set_index('userWallet')['credit_score'].to_dict()
    with open(output_path, 'w') as f:
        json.dump(scores_dict, f, indent=4)

# Main function
def main(input_file, output_file):
    df = load_data(input_file)
    features_df = engineer_features(df)
    features_df = normalize_features(features_df)
    scores_df = compute_scores(features_df)
    save_scores(scores_df, output_file)
    dist_counts = generate_distribution(scores_df)
    features_df.to_csv('features.csv', index=False)
    plt.hist(scores_df['credit_score'], bins=range(0, 1001, 100), edgecolor='black')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.title('Credit Score Distribution')
    plt.savefig('score_histogram.png')
    plt.close()
    print("Score Distribution:")
    print(scores_df['credit_score'].describe())
    print("\nFeature Correlations with Score:")
    print(features_df.corr()['credit_score'])
    print("\nScore Range Counts:")
    print(dist_counts)

if __name__ == "__main__":
    input_file = "user-wallet-transactions.json"
    output_file = "wallet_scores1.json"
    main(input_file, output_file)
'''
import pandas as pd
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Load JSON data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['amount'] = df['actionData'].apply(lambda x: float(x['amount']))
    df['assetSymbol'] = df['actionData'].apply(lambda x: x['assetSymbol'])
    df['assetPriceUSD'] = df['actionData'].apply(lambda x: float(x['assetPriceUSD']))
    df['amountUSD'] = df['amount'] * df['assetPriceUSD'] / (10 ** df['assetSymbol'].map({'USDC': 6, 'WMATIC': 18, 'DAI': 18, 'WETH': 18}).fillna(18))
    return df[['userWallet', 'action', 'timestamp', 'amountUSD', 'assetSymbol']]

# Feature engineering
def engineer_features(df):
    grouped = df.groupby('userWallet')
    features = {
        'userWallet': [],
        'total_deposits_usd': [],
        'total_borrows_usd': [],
        'repayment_ratio': [],
        'liquidation_count': [],
        'avg_transaction_size_usd': [],
        'activity_days': [],
        'token_diversity': [],
        'transaction_count': []
    }
    
    for wallet, group in grouped:
        features['userWallet'].append(wallet)
        deposits = group[group['action'] == 'deposit']['amountUSD'].sum()
        redeems = group[group['action'] == 'redeemunderlying']['amountUSD'].sum()
        features['total_deposits_usd'].append(max(deposits - redeems, 0))
        borrows = group[group['action'] == 'borrow']['amountUSD'].sum()
        features['total_borrows_usd'].append(borrows)
        repays = group[group['action'] == 'repay']['amountUSD'].sum()
        repayment_ratio = repays / borrows if borrows > 0 else 0.3  # Adjusted default
        features['repayment_ratio'].append(min(repayment_ratio, 1.0))
        liquidations = len(group[group['action'] == 'liquidationcall'])
        features['liquidation_count'].append(liquidations)
        avg_size = group['amountUSD'].mean()
        features['avg_transaction_size_usd'].append(avg_size if not np.isnan(avg_size) else 0)
        timestamps = pd.to_datetime(group['timestamp'], unit='s')
        activity_span = (timestamps.max() - timestamps.min()).days
        features['activity_days'].append(activity_span if activity_span > 0 else 1)
        features['token_diversity'].append(len(group['assetSymbol'].unique()))
        features['transaction_count'].append(len(group))
    
    return pd.DataFrame(features)

# Normalize features to [0, 1]
def normalize_features(df):
    for col in ['total_deposits_usd', 'total_borrows_usd', 'avg_transaction_size_usd', 'activity_days', 'token_diversity', 'transaction_count']:
        max_val = df[col].max()
        if max_val > 0:
            df[col] = df[col] / max_val
    max_liquidations = df['liquidation_count'].max()
    if max_liquidations > 0:
        df['liquidation_count'] = 1 - (df['liquidation_count'] / max_liquidations)
    return df

# Compute credit score
def compute_scores(features_df):
    weights = {
        'total_deposits_usd': 0.25,
        'repayment_ratio': 0.20,
        'liquidation_count': 0.25,
        'avg_transaction_size_usd': 0.10,
        'activity_days': 0.10,
        'token_diversity': 0.05,
        'transaction_count': 0.05
    }
    
    scores = (
        features_df['total_deposits_usd'] * weights['total_deposits_usd'] +
        features_df['repayment_ratio'] * weights['repayment_ratio'] +
        features_df['liquidation_count'] * weights['liquidation_count'] +
        features_df['avg_transaction_size_usd'] * weights['avg_transaction_size_usd'] +
        features_df['activity_days'] * weights['activity_days'] +
        features_df['token_diversity'] * weights['token_diversity'] +
        features_df['transaction_count'] * weights['transaction_count']
    )
    scores = 1000 * (1 / (1 + np.exp(-3 * (scores - 0.5))))  # Softer scaling
    
    features_df['credit_score'] = scores.clip(0, 1000).round().astype(int)
    return features_df[['userWallet', 'credit_score']]

# Generate score distribution
def generate_distribution(scores_df):
    bins = range(0, 1001, 100)
    labels = [f"{i}-{i+100}" for i in range(0, 1000, 100)]
    distribution = pd.cut(scores_df['credit_score'], bins=bins, labels=labels, include_lowest=True)
    dist_counts = distribution.value_counts().sort_index().to_dict()
    with open('score_distribution.json', 'w') as f:
        json.dump(dist_counts, f, indent=4)
    return dist_counts

# Save scores to JSON
def save_scores(scores_df, output_path):
    scores_dict = scores_df.set_index('userWallet')['credit_score'].to_dict()
    with open(output_path, 'w') as f:
        json.dump(scores_dict, f, indent=4)

# Main function
def main(input_file, output_file):
    df = load_data(input_file)
    features_df = engineer_features(df)
    features_df = normalize_features(features_df)
    scores_df = compute_scores(features_df)
    save_scores(scores_df, output_file)
    dist_counts = generate_distribution(scores_df)
    features_df.to_csv('features.csv', index=False)
    plt.hist(scores_df['credit_score'], bins=range(0, 1001, 100), edgecolor='black')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.title('Credit Score Distribution')
    plt.savefig('score_histogram.png')
    plt.close()
    print("Score Distribution:")
    print(scores_df['credit_score'].describe())
    print("\nFeature Correlations with Score:")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(features_df[numeric_cols].corr()['credit_score'])
    print("\nScore Range Counts:")
    print(dist_counts)

if __name__ == "__main__":
    input_file = "user-wallet-transactions.json"
    output_file = "wallet_scores.json"
    main(input_file, output_file)