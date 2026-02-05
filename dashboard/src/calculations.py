import pandas as pd
import numpy as np
from sector_data import get_sector

def clean_and_normalize_data(kse_df, kmi_df):
    """
    Merges KSE and KMI data, finding common stocks and calculating initial weights.
    """
    if kse_df is None or kmi_df is None or kse_df.empty or kmi_df.empty:
        return pd.DataFrame()

    # Ensure numeric types
    for col in ["Weight_KSE100", "Price_KSE100"]:
        if col in kse_df.columns:
            kse_df[col] = pd.to_numeric(kse_df[col], errors='coerce')
            
    for col in ["Weight_KMI30", "Price_KMI30"]:
        if col in kmi_df.columns:
            kmi_df[col] = pd.to_numeric(kmi_df[col], errors='coerce')

    # Drop NaNs
    kse_df = kse_df.dropna()
    kmi_df = kmi_df.dropna()

    # Merge on Stock
    # Assuming Stock names are consistent. If not, fuzzy matching might be needed, 
    # but for now we rely on exact match as implied by source.
    common = pd.merge(kse_df, kmi_df, on="Stock", how="inner")

    if common.empty:
        return common

    # Calculate attributes
    # Weight average
    common["Default Weight"] = (common["Weight_KSE100"] + common["Weight_KMI30"]) / 2
    
    # Average Price
    common["Price"] = (common["Price_KSE100"] + common["Price_KMI30"]) / 2
    
    # Initialize User Adjusted Weight with Default Weight
    common["Final Weight"] = common["Default Weight"]

    # Assign Sector
    common["Sector"] = common["Stock"].apply(get_sector)

    # Normalize Initial Weights to equal 100%
    total_weight = common["Final Weight"].sum()
    if total_weight > 0:
        common["Final Weight"] = (common["Final Weight"] / total_weight) * 100
        common["Default Weight"] = (common["Default Weight"] / total_weight) * 100 # Adjust default to 100% scale too for comparison

    return common.sort_values("Final Weight", ascending=False).reset_index(drop=True)

def calculate_allocation(df, total_investment, transaction_fee_pct=0.0):
    """
    Calculates the allocated amount and shares to buy based on 'Final Weight'.
    """
    if df.empty or total_investment <= 0:
        df["Allocated Amount"] = 0
        df["Shares to Buy"] = 0
        df["Transaction Cost"] = 0
        return df

    # Calculate Allocation (Gross)
    # The transaction fee is deducted from the investment amount available for that stock? 
    # Or is it an extra cost? usually it's deduced.
    # Investment = (Price * Qty) + Fee
    # Fee = (Price * Qty) * fee_pct
    # Investment = (Price * Qty) * (1 + fee_pct)
    # Price * Qty = Investment / (1 + fee_pct)
    
    # Let's say we want to spend X amount including fee.
    fee_multiplier = 1 + (transaction_fee_pct / 100)
    
    df["Allocated Amount"] = total_investment * (df["Final Weight"] / 100)
    
    # Calculate Net Amount available for shares
    df["Net Investment"] = df["Allocated Amount"] / fee_multiplier
    
    # Calculate Transaction Cost per stock
    df["Transaction Cost"] = df["Allocated Amount"] - df["Net Investment"]
    
    # Calculate Shares
    df["Shares to Buy"] = df.apply(
        lambda row: row["Net Investment"] / row["Price"] if row["Price"] > 0 else 0, 
        axis=1
    )
    
    return df


def rebalance_weights(new_df, old_df):
    """
    Rebalances weights based on user changes.
    - If a user changes a weight, that weight becomes fixed.
    - The remaining weights are scaled proportionally to fit the remaining percentage (100 - fixed_weight).
    - If a weight is set to 0, it contributes 0.
    """
    # Identify changed rows by comparing with the previous state
    # We use a small epsilon for float comparison
    diff_mask = (new_df["Final Weight"] - old_df["Final Weight"]).abs() > 0.0001
    
    # If no changes, return as is
    if not diff_mask.any():
        return new_df
        
    changed_indices = new_df[diff_mask].index
    unchanged_indices = new_df.index.difference(changed_indices)
    
    # Calculate the sum of weights that the user explicitly changed
    fixed_weight_sum = new_df.loc[changed_indices, "Final Weight"].sum()
    
    # Cap fixed sum at 100 to prevent invalid states
    if fixed_weight_sum > 100:
        # Scale down the changed items to exactly 100
        scale_down = 100 / fixed_weight_sum
        new_df.loc[changed_indices, "Final Weight"] *= scale_down
        fixed_weight_sum = 100
        
        # Since 100% is used, all other stocks must be 0
        new_df.loc[unchanged_indices, "Final Weight"] = 0
        return new_df

    # Calculate how much weight is left for the unchanged stocks
    remaining_capacity = 100 - fixed_weight_sum
    
    # Note: We take the PROPORTION from the Old DataFrame for unchanged items
    # ensuring we don't lose their relative scale
    current_unchanged_sum = old_df.loc[unchanged_indices, "Final Weight"].sum()
    
    if current_unchanged_sum == 0:
        # If the remaining stocks previously summed to 0, we can't scale them up proportionally.
        # Example: user set everything to 0 before, and now sets one to 50. 
        # The others rely on being 0. That's fine.
        new_df.loc[unchanged_indices, "Final Weight"] = 0
    else:
        # Scale factor combines the new capacity vs the old sum
        scale_factor = remaining_capacity / current_unchanged_sum
        new_df.loc[unchanged_indices, "Final Weight"] = old_df.loc[unchanged_indices, "Final Weight"] * scale_factor
        
    return new_df
