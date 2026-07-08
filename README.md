# PrimeTrade.ai - Market Sentiment and Trader Performance Analysis

## Objective
This project explores the relationship between Bitcoin market sentiment (using the Fear and Greed Index) and trader performance on Hyperliquid. The analysis aims to uncover hidden patterns and deliver actionable insights for smarter trading strategies.

## Datasets
* **`fear_greed_index.csv`**: Daily Bitcoin sentiment data (2018-02-01 to 2025-05-02).
* **`historical_data.csv`**: Historical dataset containing 211,224 trades executed by 32 Hyperliquid traders. *(Note: This file might be too large for some GitHub repositories and can be excluded or zipped if necessary).*

## Features & Analysis
The `analysis.py` script performs a full, reproducible data science workflow:
1. **Data Loading & Cleaning**: Merges daily sentiment data with historical trades.
2. **Exploratory Data Analysis**: Visualizes sentiment distribution, market volume, and index time-series.
3. **PnL Analysis by Sentiment**: Calculates win rates, total PnL, and profit factors based on Extreme Fear to Extreme Greed.
4. **Trader Performance Profiling**: Ranks traders, calculates Sharpe proxy, and plots win rate vs. volume.
5. **Sentiment Behavior Patterns**: Analyzes Long vs. Short preferences and coin selection under different sentiments.
6. **Temporal Patterns**: Heatmaps showing day-of-week and hourly trading activity.
7. **Statistical Tests**: Uses ANOVA, Pearson, and Spearman correlations to determine significance.
8. **Contrarian Trading Analysis**: Compares the profitability of contrarian strategies (buying in fear, selling in greed) versus trend following.
9. **Top vs. Bottom Traders**: Contrasts the habits of the top 5 most profitable traders against the bottom 5.

## How to Run

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn
   ```
3. Execute the analysis script:
   ```bash
   python analysis.py
   ```

## Outputs
The script runs headlessly (without a GUI) and automatically generates an `outputs/` directory containing:
* **17 PNG Charts**: Visualizations including sentiment distribution, hourly activity, cumulative PnL, win rate heatmaps, and more.
* **6 CSV Files**: Summaries like `trader_profile.csv`, `statistical_tests.csv`, and `contrarian_analysis.csv`.
* **`key_insights.txt`**: A text file summarizing the core findings of the analysis (e.g., best sentiment for average PnL, profitable trader count, best strategy).

## Key Insights
* Check the generated `outputs/key_insights.txt` to find out whether buying in fear or selling in greed proved to be the most profitable strategy across the dataset, as well as the significance of market sentiment on net PnL.
