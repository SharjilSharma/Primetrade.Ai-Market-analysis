"""
=============================================================================
PrimeTrade.ai Data Science Assignment - Full Reproducible Analysis
=============================================================================
Objective : Explore the relationship between Bitcoin market sentiment (Fear and
            Greed Index) and trader performance on Hyperliquid, uncover hidden
            patterns, and deliver actionable insights for smarter trading.

Datasets  : fear_greed_index.csv  -- daily BTC sentiment (2018-02-01 to 2025-05-02)
            historical_data.csv   -- 211,224 trades by 32 Hyperliquid traders

Output    : outputs/ directory with CSV summaries + PNG charts

HOW TO RUN:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
    python analysis.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering - no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#bcbd22",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#1f77b4",
}
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#161625",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a40",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1e1e30",
    "legend.edgecolor": "#444",
    "font.family":      "DejaVu Sans",
})

# ===========================================================================
# SECTION 1 -- DATA LOADING AND CLEANING
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 1 -- DATA LOADING AND CLEANING")
print("=" * 60)

fg_raw = pd.read_csv("fear_greed_index.csv")
hd_raw = pd.read_csv("historical_data.csv")

# -- Fear and Greed ----------------------------------------------------------
fg = fg_raw.copy()
fg["date"] = pd.to_datetime(fg["date"])
fg = fg.rename(columns={"classification": "sentiment", "value": "fg_value"})
fg["sentiment"] = pd.Categorical(fg["sentiment"], categories=SENTIMENT_ORDER, ordered=True)
fg = fg.sort_values("date").reset_index(drop=True)

# -- Historical Trades -------------------------------------------------------
hd = hd_raw.copy()
hd["Timestamp IST"] = pd.to_datetime(hd["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
hd["date"] = hd["Timestamp IST"].dt.normalize()
hd["net_pnl"] = hd["Closed PnL"] - hd["Fee"]   # true net profit after fees
hd["hour"] = hd["Timestamp IST"].dt.hour
hd["weekday"] = hd["Timestamp IST"].dt.day_name()
hd["month"] = hd["Timestamp IST"].dt.to_period("M")
hd["is_long"] = hd["Direction"].str.contains("Long", na=False)
hd["is_close"] = hd["Direction"].str.startswith("Close", na=False)

# Keep only closing trades for PnL analysis (open trades have PnL = 0)
closed = hd[hd["Closed PnL"] != 0].copy()

print(f"Fear and Greed records : {len(fg):,}  ({fg['date'].min().date()} to {fg['date'].max().date()})")
print(f"All trade records      : {len(hd):,}")
print(f"Closed-trade records   : {len(closed):,}  (PnL != 0)")
print(f"Unique traders         : {hd['Account'].nunique()}")
print(f"Unique coins           : {hd['Coin'].nunique()}")
print(f"Sentiment distribution : {dict(fg['sentiment'].value_counts())}")

# ===========================================================================
# SECTION 2 -- MERGE TRADES WITH SENTIMENT
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 2 -- MERGING DATASETS")
print("=" * 60)

merged        = hd.merge(fg[["date", "fg_value", "sentiment"]], on="date", how="left")
merged_closed = closed.merge(fg[["date", "fg_value", "sentiment"]], on="date", how="left")

print(f"Merged all trades    : {len(merged):,}  |  sentiment coverage: {merged['sentiment'].notna().mean()*100:.1f}%")
print(f"Merged closed trades : {len(merged_closed):,}  |  sentiment coverage: {merged_closed['sentiment'].notna().mean()*100:.1f}%")

m  = merged.dropna(subset=["sentiment"]).copy()
mc = merged_closed.dropna(subset=["sentiment"]).copy()

# ===========================================================================
# SECTION 3 -- EXPLORATORY DATA ANALYSIS
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 3 -- EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Chart 1: Sentiment Distribution
fig, ax = plt.subplots(figsize=(9, 5))
counts = fg["sentiment"].value_counts()[SENTIMENT_ORDER]
bars = ax.bar(SENTIMENT_ORDER, counts.values,
              color=[PALETTE[s] for s in SENTIMENT_ORDER], edgecolor="#000", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{val:,}", ha="center", va="bottom", fontsize=9, color="#ccc")
ax.set_title("Bitcoin Market Sentiment Distribution (2018-2025)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Sentiment Category")
ax.set_ylabel("Number of Days")
ax.grid(axis="y")
ax.set_ylim(0, counts.max() * 1.12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_sentiment_distribution.png", dpi=150)
plt.close()
print("  Chart 1 saved: 01_sentiment_distribution.png")

# Chart 2: FG Index Time-Series
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(fg["date"], fg["fg_value"], alpha=0.3, color="#4fa3e3")
ax.plot(fg["date"], fg["fg_value"].rolling(30).mean(), color="#e3a44f",
        linewidth=1.8, label="30-day MA")
ax.axhline(50, color="#888", linestyle="--", linewidth=0.8, label="Neutral (50)")
ax.set_title("Bitcoin Fear and Greed Index Over Time", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Date")
ax.set_ylabel("Fear and Greed Value (0-100)")
ax.legend(fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_fg_timeseries.png", dpi=150)
plt.close()
print("  Chart 2 saved: 02_fg_timeseries.png")

# Chart 3: Volume by Sentiment
vol_by_sent = (
    m.groupby("sentiment", observed=True)["Size USD"]
     .agg(["count", "sum", "mean"])
     .reindex(SENTIMENT_ORDER)
)
vol_by_sent.columns = ["trade_count", "total_usd", "avg_usd"]
vol_by_sent.to_csv(f"{OUTPUT_DIR}/trade_volume_by_sentiment.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
clrs = [PALETTE[s] for s in SENTIMENT_ORDER]
axes[0].bar(SENTIMENT_ORDER, vol_by_sent["trade_count"] / 1000, color=clrs, edgecolor="#000")
axes[0].set_title("Trade Count per Sentiment (000s)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Trades (thousands)")
axes[0].tick_params(axis="x", rotation=20)
axes[0].grid(axis="y")
axes[1].bar(SENTIMENT_ORDER, vol_by_sent["avg_usd"] / 1000, color=clrs, edgecolor="#000")
axes[1].set_title("Average Trade Size (USD 000s) per Sentiment", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Avg Size USD (thousands)")
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(axis="y")
plt.suptitle("Trading Activity by Market Sentiment", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_volume_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 3 saved: 03_volume_by_sentiment.png")

# ===========================================================================
# SECTION 4 -- PnL ANALYSIS BY SENTIMENT
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 4 -- PnL ANALYSIS BY SENTIMENT")
print("=" * 60)

pnl_stats = (
    mc.groupby("sentiment", observed=True)["net_pnl"]
      .agg(
          trades     = "count",
          total_pnl  = "sum",
          mean_pnl   = "mean",
          median_pnl = "median",
          std_pnl    = "std",
          win_rate   = lambda x: (x > 0).mean(),
          max_win    = "max",
          max_loss   = "min",
      )
      .reindex(SENTIMENT_ORDER)
)

pnl_stats["profit_factor"] = pnl_stats.apply(
    lambda r: mc[mc["sentiment"] == r.name]["net_pnl"].clip(lower=0).sum()
              / abs(mc[mc["sentiment"] == r.name]["net_pnl"].clip(upper=0).sum() + 1e-9),
    axis=1,
)
pnl_stats.to_csv(f"{OUTPUT_DIR}/pnl_stats_by_sentiment.csv")
print(pnl_stats[["trades", "total_pnl", "mean_pnl", "win_rate", "profit_factor"]].round(3).to_string())

# Chart 4: Win Rate by Sentiment
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(SENTIMENT_ORDER, pnl_stats["win_rate"] * 100,
              color=[PALETTE[s] for s in SENTIMENT_ORDER], edgecolor="#000")
for bar, val in zip(bars, pnl_stats["win_rate"].values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
ax.axhline(50, color="#aaa", linestyle="--", linewidth=0.9, label="50% baseline")
ax.set_title("Win Rate by Market Sentiment", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Win Rate (%)")
ax.set_ylim(0, 75)
ax.legend()
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_winrate_by_sentiment.png", dpi=150)
plt.close()
print("  Chart 4 saved: 04_winrate_by_sentiment.png")

# Chart 5: PnL Box Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_data = [mc[mc["sentiment"] == s]["net_pnl"].clip(-500, 1000) for s in SENTIMENT_ORDER]
bp = ax.boxplot(
    plot_data, tick_labels=SENTIMENT_ORDER, patch_artist=True, notch=True,
    medianprops={"color": "white", "linewidth": 2},
    whiskerprops={"color": "#aaa"},
    capprops={"color": "#aaa"},
    flierprops={"marker": ".", "color": "#aaa", "alpha": 0.3, "markersize": 3},
)
for patch, color in zip(bp["boxes"], [PALETTE[s] for s in SENTIMENT_ORDER]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, color="#aaa", linestyle="--", linewidth=0.8)
ax.set_title("Net PnL Distribution by Market Sentiment (clipped -$500 to $1k)", fontsize=12, fontweight="bold")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Net PnL (USD)")
ax.grid(axis="y")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_pnl_boxplot_by_sentiment.png", dpi=150)
plt.close()
print("  Chart 5 saved: 05_pnl_boxplot_by_sentiment.png")

# Chart 6: Total PnL by Sentiment
fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in pnl_stats["total_pnl"]]
ax.bar(SENTIMENT_ORDER, pnl_stats["total_pnl"] / 1000, color=colors_bar, edgecolor="#000")
ax.axhline(0, color="#aaa", linewidth=0.8)
ax.set_title("Total Net PnL by Sentiment (USD thousands)", fontsize=12, fontweight="bold")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Total Net PnL ($000s)")
ax.grid(axis="y")
ax.tick_params(axis="x", rotation=15)
for i, (s, v) in enumerate(zip(SENTIMENT_ORDER, pnl_stats["total_pnl"])):
    ax.text(i, v / 1000 + (2 if v >= 0 else -5), f"${v/1000:.0f}k",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_total_pnl_by_sentiment.png", dpi=150)
plt.close()
print("  Chart 6 saved: 06_total_pnl_by_sentiment.png")

# ===========================================================================
# SECTION 5 -- TRADER PERFORMANCE PROFILING
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 5 -- TRADER PERFORMANCE PROFILING")
print("=" * 60)

trader_stats = (
    mc.groupby("Account")
      .agg(
          trades       = ("net_pnl", "count"),
          total_pnl    = ("net_pnl", "sum"),
          mean_pnl     = ("net_pnl", "mean"),
          win_rate     = ("net_pnl", lambda x: (x > 0).mean()),
          total_volume = ("Size USD", "sum"),
          avg_size     = ("Size USD", "mean"),
          max_loss     = ("net_pnl", "min"),
          max_win      = ("net_pnl", "max"),
      )
      .reset_index()
)

std_by_account = mc.groupby("Account")["net_pnl"].std().reset_index()
std_by_account.columns = ["Account", "pnl_std"]
trader_stats = trader_stats.merge(std_by_account, on="Account")
trader_stats["sharpe_proxy"] = trader_stats["mean_pnl"] / (trader_stats["pnl_std"] + 1e-9)
trader_stats["profitability"] = np.where(trader_stats["total_pnl"] > 0, "Profitable", "Unprofitable")
trader_stats = trader_stats.sort_values("total_pnl", ascending=False).reset_index(drop=True)
trader_stats.to_csv(f"{OUTPUT_DIR}/trader_profile.csv", index=False)

n_profitable = (trader_stats["total_pnl"] > 0).sum()
print(f"  Profitable traders : {n_profitable} / {len(trader_stats)}")
print(f"  Best trader PnL    : ${trader_stats['total_pnl'].max():,.0f}")
print(f"  Worst trader PnL   : ${trader_stats['total_pnl'].min():,.0f}")

# Chart 7: Trader PnL Ranking
fig, ax = plt.subplots(figsize=(12, 6))
short_labels = [f"T{i+1}" for i in range(len(trader_stats))]
bar_colors = ["#2ca02c" if v > 0 else "#d62728" for v in trader_stats["total_pnl"]]
ax.bar(short_labels, trader_stats["total_pnl"] / 1000, color=bar_colors, edgecolor="#000")
ax.axhline(0, color="#aaa", linewidth=0.8)
ax.set_title("Total Net PnL per Trader (USD thousands, T1=Best)", fontsize=12, fontweight="bold")
ax.set_xlabel("Trader")
ax.set_ylabel("Total Net PnL ($000s)")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_trader_pnl_ranking.png", dpi=150)
plt.close()
print("  Chart 7 saved: 07_trader_pnl_ranking.png")

# Chart 8: Win Rate vs Volume Scatter
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(
    trader_stats["total_volume"] / 1e6,
    trader_stats["win_rate"] * 100,
    c=trader_stats["total_pnl"],
    cmap="RdYlGn", s=80, edgecolor="white", linewidth=0.5,
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Total Net PnL ($)", color="#ccc")
ax.set_title("Win Rate vs. Total Volume -- Trader Scatter", fontsize=12, fontweight="bold")
ax.set_xlabel("Total Volume ($ millions)")
ax.set_ylabel("Win Rate (%)")
ax.axhline(50, color="#aaa", linestyle="--", linewidth=0.8)
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_winrate_vs_volume_scatter.png", dpi=150)
plt.close()
print("  Chart 8 saved: 08_winrate_vs_volume_scatter.png")

# ===========================================================================
# SECTION 6 -- SENTIMENT BEHAVIOUR PATTERNS
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 6 -- SENTIMENT BEHAVIOUR PATTERNS")
print("=" * 60)

# Chart 9: Long vs Short by Sentiment
direction_sent = (
    m.groupby(["sentiment", "Direction"], observed=True)
     .size()
     .unstack(fill_value=0)
)
ls = pd.DataFrame({
    "Long":        direction_sent.get("Open Long",   0),
    "Close Long":  direction_sent.get("Close Long",  0),
    "Short":       direction_sent.get("Open Short",  0),
    "Close Short": direction_sent.get("Close Short", 0),
}).reindex(SENTIMENT_ORDER).fillna(0)
ls["Long_pct"] = ls["Long"] / (ls["Long"] + ls["Short"] + 1e-9) * 100
ls.to_csv(f"{OUTPUT_DIR}/long_short_ratio_by_sentiment.csv")

x = np.arange(len(SENTIMENT_ORDER))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width / 2, ls["Long"],  width, label="Open Long",  color="#2ca02c", edgecolor="#000")
ax.bar(x + width / 2, ls["Short"], width, label="Open Short", color="#d62728", edgecolor="#000")
ax.set_xticks(x)
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15)
ax.set_title("Long vs Short Openings by Sentiment", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Trades")
ax.legend()
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_long_short_by_sentiment.png", dpi=150)
plt.close()
print("  Chart 9 saved: 09_long_short_by_sentiment.png")
print(ls[["Long", "Short", "Long_pct"]].round(1).to_string())

# Chart 10: Coin Preference by Sentiment
top_coins = m["Coin"].value_counts().head(6).index.tolist()
coin_sent = (
    m[m["Coin"].isin(top_coins)]
     .groupby(["sentiment", "Coin"], observed=True)["Size USD"]
     .sum()
     .unstack(fill_value=0)
     .reindex(SENTIMENT_ORDER)
)
coin_sent_pct = coin_sent.div(coin_sent.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 6))
coin_sent_pct.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="#000")
ax.set_title("Coin Preference (% of Volume) by Sentiment", fontsize=12, fontweight="bold")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Volume Share (%)")
ax.legend(title="Coin", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_coin_preference_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 10 saved: 10_coin_preference_by_sentiment.png")

# ===========================================================================
# SECTION 7 -- TEMPORAL PATTERNS
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 7 -- TEMPORAL PATTERNS")
print("=" * 60)

# Chart 11: PnL Heatmap Day-of-Week x Sentiment
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
mc["weekday"] = mc["Timestamp IST"].dt.day_name()
dow_sent = (
    mc.groupby(["weekday", "sentiment"], observed=True)["net_pnl"]
      .mean()
      .unstack(fill_value=0)
      .reindex(DOW_ORDER)
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(dow_sent, annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=ax,
            linewidths=0.5, linecolor="#333")
ax.set_title("Avg Net PnL Heatmap: Day of Week x Sentiment", fontsize=12, fontweight="bold")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Day of Week")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_pnl_heatmap_dow_sentiment.png", dpi=150)
plt.close()
print("  Chart 11 saved: 11_pnl_heatmap_dow_sentiment.png")

# Chart 12: Hourly Trade Activity
mc["hour"] = mc["Timestamp IST"].dt.hour
hourly = mc.groupby("hour")["net_pnl"].agg(["count", "mean"]).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.bar(hourly["hour"], hourly["count"], color="#4fa3e3", alpha=0.6, label="Trade Count")
ax2.plot(hourly["hour"], hourly["mean"], color="#ff7f0e", linewidth=2,
         marker="o", markersize=4, label="Avg PnL")
ax2.axhline(0, color="#aaa", linestyle="--", linewidth=0.8)
ax1.set_title("Hourly Trading Activity and Average PnL (IST)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Hour (IST)")
ax1.set_ylabel("Trade Count", color="#4fa3e3")
ax2.set_ylabel("Avg Net PnL ($)", color="#ff7f0e")
ax1.set_xticks(range(24))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_hourly_activity.png", dpi=150)
plt.close()
print("  Chart 12 saved: 12_hourly_activity.png")

# Chart 13: Monthly Cumulative PnL
mc["month_ts"] = mc["Timestamp IST"].dt.to_period("M").dt.to_timestamp()
monthly_pnl     = mc.groupby("month_ts")["net_pnl"].sum().sort_index()
monthly_pnl_cum = monthly_pnl.cumsum()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(monthly_pnl_cum.index, monthly_pnl_cum.values / 1000, alpha=0.3, color="#4fa3e3")
ax.plot(monthly_pnl_cum.index, monthly_pnl_cum.values / 1000, color="#4fa3e3", linewidth=1.8)
ax.axhline(0, color="#aaa", linestyle="--", linewidth=0.8)
ax.set_title("Cumulative Net PnL Across All Traders (USD thousands)", fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Cumulative PnL ($000s)")
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_cumulative_pnl.png", dpi=150)
plt.close()
print("  Chart 13 saved: 13_cumulative_pnl.png")

# ===========================================================================
# SECTION 8 -- STATISTICAL TESTS
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 8 -- STATISTICAL TESTS")
print("=" * 60)

# ANOVA: PnL across sentiment groups
groups   = [mc[mc["sentiment"] == s]["net_pnl"].dropna() for s in SENTIMENT_ORDER]
f_stat, p_val = stats.f_oneway(*groups)
print(f"  One-Way ANOVA (PnL ~ Sentiment): F={f_stat:.3f}, p={p_val:.4f}")
sig_anova = p_val < 0.05
print(f"  {'SIGNIFICANT' if sig_anova else 'NOT SIGNIFICANT'} at alpha=0.05")

# Pearson Correlation: FG value vs daily avg PnL
daily_pnl = mc.groupby("date")["net_pnl"].mean().reset_index()
daily_pnl = daily_pnl.merge(fg[["date", "fg_value"]], on="date", how="inner")
r, p_r = stats.pearsonr(daily_pnl["fg_value"], daily_pnl["net_pnl"])
print(f"\n  Pearson r (FG value vs daily avg PnL): r={r:.4f}, p={p_r:.4f}")
sig_pearson = p_r < 0.05
print(f"  {'SIGNIFICANT' if sig_pearson else 'NOT SIGNIFICANT'} at alpha=0.05")

# Spearman Correlation
rho, p_s = stats.spearmanr(daily_pnl["fg_value"], daily_pnl["net_pnl"])
print(f"  Spearman rho                          : rho={rho:.4f}, p={p_s:.4f}")

# Chart 14: Scatter FG vs daily PnL
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(daily_pnl["fg_value"], daily_pnl["net_pnl"],
           alpha=0.4, s=20, color="#4fa3e3", edgecolors="none")
m_ols, b_ols = np.polyfit(daily_pnl["fg_value"], daily_pnl["net_pnl"], 1)
x_line = np.linspace(daily_pnl["fg_value"].min(), daily_pnl["fg_value"].max(), 100)
ax.plot(x_line, m_ols * x_line + b_ols, color="#ff7f0e", linewidth=2, label=f"OLS (r={r:.3f})")
ax.set_title("Fear and Greed Index vs Daily Average Net PnL", fontsize=12, fontweight="bold")
ax.set_xlabel("Fear and Greed Value (0=Fear, 100=Greed)")
ax.set_ylabel("Daily Avg Net PnL ($)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_fg_vs_pnl_scatter.png", dpi=150)
plt.close()
print("  Chart 14 saved: 14_fg_vs_pnl_scatter.png")

stats_results = pd.DataFrame({
    "test":        ["ANOVA (PnL ~ Sentiment)", "Pearson r (FG vs daily PnL)", "Spearman rho (FG vs daily PnL)"],
    "statistic":   [f_stat, r, rho],
    "p_value":     [p_val, p_r, p_s],
    "significant": [sig_anova, sig_pearson, p_s < 0.05],
})
stats_results.to_csv(f"{OUTPUT_DIR}/statistical_tests.csv", index=False)

# ===========================================================================
# SECTION 9 -- CONTRARIAN TRADING ANALYSIS
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 9 -- CONTRARIAN TRADING ANALYSIS")
print("=" * 60)

mc["contrarian_buy"]  = (mc["Side"] == "BUY")  & mc["sentiment"].isin(["Extreme Fear", "Fear"])
mc["contrarian_sell"] = (mc["Side"] == "SELL") & mc["sentiment"].isin(["Extreme Greed", "Greed"])

contra_buy_pnl  = mc[mc["contrarian_buy"]]["net_pnl"].mean()
contra_sell_pnl = mc[mc["contrarian_sell"]]["net_pnl"].mean()
trend_buy_pnl   = mc[(mc["Side"] == "BUY")  & mc["sentiment"].isin(["Extreme Greed", "Greed"])]["net_pnl"].mean()
trend_sell_pnl  = mc[(mc["Side"] == "SELL") & mc["sentiment"].isin(["Extreme Fear", "Fear"])]["net_pnl"].mean()

contrarian_df = pd.DataFrame({
    "Strategy":      ["Contrarian Buy (Buy in Fear)", "Trend Buy (Buy in Greed)",
                      "Contrarian Sell (Sell in Greed)", "Trend Sell (Sell in Fear)"],
    "Avg Net PnL":   [contra_buy_pnl, trend_buy_pnl, contra_sell_pnl, trend_sell_pnl],
})
contrarian_df.to_csv(f"{OUTPUT_DIR}/contrarian_analysis.csv", index=False)
print(contrarian_df.to_string(index=False))

# Chart 15: Contrarian vs Trend
fig, ax = plt.subplots(figsize=(9, 5))
colors_c = ["#2ca02c" if v > 0 else "#d62728" for v in contrarian_df["Avg Net PnL"]]
ax.bar(contrarian_df["Strategy"], contrarian_df["Avg Net PnL"], color=colors_c, edgecolor="#000")
ax.axhline(0, color="#aaa", linewidth=0.8)
ax.set_title("Contrarian vs Trend Following: Avg Net PnL", fontsize=12, fontweight="bold")
ax.set_ylabel("Avg Net PnL ($)")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/15_contrarian_analysis.png", dpi=150)
plt.close()
print("  Chart 15 saved: 15_contrarian_analysis.png")

# ===========================================================================
# SECTION 10 -- TOP vs BOTTOM TRADER COMPARISON
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 10 -- TOP vs BOTTOM TRADER COMPARISON")
print("=" * 60)

top5_accounts = trader_stats.head(5)["Account"].tolist()
bot5_accounts = trader_stats.tail(5)["Account"].tolist()

top_mc = mc[mc["Account"].isin(top5_accounts)].copy()
bot_mc = mc[mc["Account"].isin(bot5_accounts)].copy()

top_sent_pnl = top_mc.groupby("sentiment", observed=True)["net_pnl"].mean().reindex(SENTIMENT_ORDER)
bot_sent_pnl = bot_mc.groupby("sentiment", observed=True)["net_pnl"].mean().reindex(SENTIMENT_ORDER)

# Chart 16: Top vs Bottom by Sentiment
x = np.arange(len(SENTIMENT_ORDER))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width / 2, top_sent_pnl.values, width, label="Top 5 Traders",    color="#2ca02c", edgecolor="#000")
ax.bar(x + width / 2, bot_sent_pnl.values, width, label="Bottom 5 Traders", color="#d62728", edgecolor="#000")
ax.set_xticks(x)
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15)
ax.axhline(0, color="#aaa", linewidth=0.8)
ax.set_title("Avg Net PnL: Top 5 vs Bottom 5 Traders by Sentiment", fontsize=12, fontweight="bold")
ax.set_ylabel("Avg Net PnL ($)")
ax.legend()
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/16_top_vs_bottom_traders.png", dpi=150)
plt.close()
print("  Chart 16 saved: 16_top_vs_bottom_traders.png")

# Chart 17: Win Rate Comparison Heatmap
top_wr = top_mc.groupby("sentiment", observed=True)["net_pnl"].apply(lambda x: (x > 0).mean()).reindex(SENTIMENT_ORDER)
bot_wr = bot_mc.groupby("sentiment", observed=True)["net_pnl"].apply(lambda x: (x > 0).mean()).reindex(SENTIMENT_ORDER)

wr_df = pd.DataFrame({"Top 5": top_wr.values * 100, "Bottom 5": bot_wr.values * 100},
                     index=SENTIMENT_ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(wr_df, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax,
            linewidths=0.5, linecolor="#333")
ax.set_title("Win Rate (%) Heatmap: Top vs Bottom Traders by Sentiment", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/17_winrate_heatmap_top_bottom.png", dpi=150)
plt.close()
print("  Chart 17 saved: 17_winrate_heatmap_top_bottom.png")

# ===========================================================================
# SECTION 11 -- KEY INSIGHTS SUMMARY
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 11 -- KEY INSIGHTS SUMMARY")
print("=" * 60)

best_sent_pnl  = pnl_stats["mean_pnl"].idxmax()
worst_sent_pnl = pnl_stats["mean_pnl"].idxmin()
best_wr_sent   = pnl_stats["win_rate"].idxmax()
best_strategy  = contrarian_df.loc[contrarian_df["Avg Net PnL"].idxmax(), "Strategy"]
n_fear_days    = int(fg["sentiment"].value_counts().get("Fear", 0) + fg["sentiment"].value_counts().get("Extreme Fear", 0))

insights = [
    f"1. BEST sentiment for avg PnL      : {best_sent_pnl} (avg ${pnl_stats.loc[best_sent_pnl,'mean_pnl']:.2f})",
    f"2. WORST sentiment for avg PnL     : {worst_sent_pnl} (avg ${pnl_stats.loc[worst_sent_pnl,'mean_pnl']:.2f})",
    f"3. BEST win rate sentiment         : {best_wr_sent} ({pnl_stats.loc[best_wr_sent,'win_rate']*100:.1f}%)",
    f"4. ANOVA significance              : F={f_stat:.3f}, p={p_val:.4f} ({'SIG' if sig_anova else 'NOT SIG'})",
    f"5. Pearson r (FG value vs PnL)     : {r:.4f} ({'sig' if sig_pearson else 'not sig'})",
    f"6. Profitable traders              : {n_profitable} / {len(trader_stats)}",
    f"7. Best strategy (contrarian)      : {best_strategy}",
    f"8. Fear-dominated market days      : {n_fear_days} / {len(fg)} ({n_fear_days/len(fg)*100:.1f}%)",
]

for ins in insights:
    print("  " + ins)

with open(f"{OUTPUT_DIR}/key_insights.txt", "w", encoding="utf-8") as fh:
    fh.write("PrimeTrade.ai Analysis -- Key Insights\n")
    fh.write("=" * 50 + "\n")
    for ins in insights:
        fh.write(ins + "\n")

print(f"\n{'='*60}")
print("ALL ANALYSIS COMPLETE!")
print(f"  Output directory: ./{OUTPUT_DIR}/")
print(f"  Charts: 17 PNG files")
print(f"  Data  : 6 CSV files + key_insights.txt")
print(f"{'='*60}\n")
