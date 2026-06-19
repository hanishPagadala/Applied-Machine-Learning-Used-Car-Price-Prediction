import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("cleaneddatasheet.csv")

df = df[["price", "miles", "carAge"]].copy()

df = df[(df["price"] > 0) & (df["miles"] > 0) & (df["carAge"] > 0)]

print("Dataset shape:", df.shape)

money_format = FuncFormatter(lambda value, position: f"${value:,.0f}")

plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=40)
plt.xscale("log")
plt.gca().xaxis.set_major_formatter(money_format)

plt.title("Price Distribution (Log Scale)")
plt.xlabel("Price (log scale)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("results/plots/price_distribution_log.png", dpi=300, bbox_inches="tight")
plt.close()

sample_df = df.sample(n=min(3000, len(df)), random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(sample_df["carAge"], sample_df["price"], alpha=0.2, s=5)
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(money_format)

plt.title("Price vs Car Age (Log Price Scale)")
plt.xlabel("Car Age")
plt.ylabel("Price (log scale)")

plt.tight_layout()
plt.savefig("results/plots/price_vs_car_age_log.png", dpi=300, bbox_inches="tight")
plt.close()

mileage_bins = [0, 10000, 25000, 50000, 100000, 150000, 200000, float("inf")]

mileage_labels = [
    "0-10,000",
    "10,000-25,000",
    "25,000-50,000",
    "50,000-100,000",
    "100,000-150,000",
    "150,000-200,000",
    "200,000+"
]

df["Mileage Range"] = pd.cut(
    df["miles"],
    bins=mileage_bins,
    labels=mileage_labels,
    right=False
)

mileage_summary = (
    df.groupby("Mileage Range", observed=False)["price"]
    .agg(["median", "count"])
    .reset_index()
)

mileage_summary.columns = ["Mileage Range", "Median Price", "Vehicle Count"]

plt.figure(figsize=(10, 6))
plt.bar(mileage_summary["Mileage Range"].astype(str), mileage_summary["Median Price"])
plt.gca().yaxis.set_major_formatter(money_format)

plt.title("Median Price by Mileage Range")
plt.xlabel("Mileage Range")
plt.ylabel("Median Price")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig("results/plots/median_price_by_mileage_range.png", dpi=300, bbox_inches="tight")
plt.close()