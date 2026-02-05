"""
Advanced Technical Indicators for Stock Analysis
=================================================
This module provides advanced technical analysis tools including:
1. Dynamic Support & Resistance Zones (Clustering Method)
2. Geometric Pattern Recognition (Head & Shoulders, Double Bottom, etc.)

Dependencies: pandas, numpy, scipy
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Tuple, List, Optional, Dict


class SupportResistanceDetector:
    """
    Detects dynamic support and resistance zones using pivot point clustering.
    
    The algorithm:
    1. Identifies local maxima (peaks) and minima (valleys) using scipy.signal.argrelextrema
    2. Clusters nearby pivot points within a tolerance percentage
    3. Returns the most relevant support/resistance zones relative to current price
    
    Parameters:
    -----------
    window : int
        Number of candles on each side to compare for finding local extrema (default: 5)
    tolerance_pct : float
        Percentage tolerance for clustering nearby pivots (default: 1.5%)
        If two pivots are within this % of each other, they form a zone
    min_touches : int
        Minimum number of touches to consider a zone significant (default: 2)
    """
    
    def __init__(self, window: int = 5, tolerance_pct: float = 1.5, min_touches: int = 2):
        self.window = window
        self.tolerance_pct = tolerance_pct / 100  # Convert to decimal
        self.min_touches = min_touches
    
    def find_pivot_points(self, df: pd.DataFrame, price_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Find local maxima (resistance candidates) and minima (support candidates).
        
        Uses scipy.signal.argrelextrema which compares each point to its neighbors
        within the specified window. A point is a local max if it's greater than
        all points within 'window' candles on both sides.
        
        Returns:
        --------
        Tuple of (peak_indices, valley_indices)
        """
        prices = df[price_col].values
        
        # Find local maxima (peaks) - potential resistance levels
        # argrelextrema returns indices where the value is greater than neighbors
        peak_indices = argrelextrema(prices, np.greater, order=self.window)[0]
        
        # Find local minima (valleys) - potential support levels
        valley_indices = argrelextrema(prices, np.less, order=self.window)[0]
        
        return peak_indices, valley_indices
    
    def cluster_levels(self, prices: np.ndarray) -> List[Dict]:
        """
        Cluster nearby price levels into zones.
        
        The clustering logic:
        1. Sort all pivot prices
        2. Iterate through sorted prices
        3. If current price is within tolerance_pct of the previous cluster's mean,
           add it to that cluster
        4. Otherwise, start a new cluster
        
        Tolerance Calculation:
        ----------------------
        For a cluster with mean price P, a new price P_new joins if:
            |P_new - P| / P <= tolerance_pct
        
        This means a 1.5% tolerance at price 100 allows prices from 98.5 to 101.5
        to be grouped together.
        
        Returns:
        --------
        List of dicts with 'level' (zone center), 'touches' (count), 'prices' (all prices in zone)
        """
        if len(prices) == 0:
            return []
        
        # Sort prices for sequential clustering
        sorted_prices = np.sort(prices)
        
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for i in range(1, len(sorted_prices)):
            price = sorted_prices[i]
            cluster_mean = np.mean(current_cluster)
            
            # Check if price is within tolerance of current cluster mean
            # Tolerance is calculated as: |price - mean| / mean <= tolerance_pct
            if abs(price - cluster_mean) / cluster_mean <= self.tolerance_pct:
                current_cluster.append(price)
            else:
                # Save current cluster and start new one
                clusters.append({
                    'level': np.mean(current_cluster),
                    'touches': len(current_cluster),
                    'prices': current_cluster.copy()
                })
                current_cluster = [price]
        
        # Don't forget the last cluster
        clusters.append({
            'level': np.mean(current_cluster),
            'touches': len(current_cluster),
            'prices': current_cluster.copy()
        })
        
        # Filter by minimum touches
        clusters = [c for c in clusters if c['touches'] >= self.min_touches]
        
        return clusters
    
    def get_zones(self, df: pd.DataFrame, price_col: str = 'Close') -> Tuple[List[Dict], List[Dict]]:
        """
        Get all support and resistance zones.
        
        Returns:
        --------
        Tuple of (support_zones, resistance_zones)
        Each zone is a dict with 'level', 'touches', 'strength'
        """
        peak_indices, valley_indices = self.find_pivot_points(df, price_col)
        
        prices = df[price_col].values
        
        # Get prices at pivot points
        peak_prices = prices[peak_indices] if len(peak_indices) > 0 else np.array([])
        valley_prices = prices[valley_indices] if len(valley_indices) > 0 else np.array([])
        
        # Cluster into zones
        resistance_zones = self.cluster_levels(peak_prices)
        support_zones = self.cluster_levels(valley_prices)
        
        # Add strength score (more touches = stronger zone)
        for zone in resistance_zones:
            zone['strength'] = min(zone['touches'] / 5, 1.0)  # Normalize to 0-1
            zone['type'] = 'resistance'
            
        for zone in support_zones:
            zone['strength'] = min(zone['touches'] / 5, 1.0)
            zone['type'] = 'support'
        
        return support_zones, resistance_zones
    
    def get_nearest_levels(self, df: pd.DataFrame, price_col: str = 'Close', 
                           n_levels: int = 3) -> Dict:
        """
        Get the nearest support and resistance levels relative to current price.
        
        Parameters:
        -----------
        df : DataFrame with OHLCV data
        price_col : Column name for price (default: 'Close')
        n_levels : Number of nearest levels to return on each side
        
        Returns:
        --------
        Dict with:
            'current_price': float
            'support_levels': list of nearest support zones below current price
            'resistance_levels': list of nearest resistance zones above current price
        """
        current_price = df[price_col].iloc[-1]
        support_zones, resistance_zones = self.get_zones(df, price_col)
        
        # Filter and sort supports (below current price)
        supports = [z for z in support_zones if z['level'] < current_price]
        supports = sorted(supports, key=lambda x: x['level'], reverse=True)[:n_levels]
        
        # Filter and sort resistances (above current price)
        resistances = [z for z in resistance_zones if z['level'] > current_price]
        resistances = sorted(resistances, key=lambda x: x['level'])[:n_levels]
        
        return {
            'current_price': current_price,
            'support_levels': supports,
            'resistance_levels': resistances
        }


class PatternRecognizer:
    """
    Recognizes geometric chart patterns using pivot point sequences.
    
    Patterns detected:
    - Head and Shoulders (bearish reversal)
    - Inverse Head and Shoulders (bullish reversal)
    - Double Bottom (bullish reversal)
    - Double Top (bearish reversal)
    - Triple Bottom (bullish reversal)
    - Triple Top (bearish reversal)
    
    Parameters:
    -----------
    window : int
        Window for detecting pivot points (default: 5)
    tolerance_pct : float
        Price tolerance for matching levels (e.g., shoulders must be within this % of each other)
    """
    
    def __init__(self, window: int = 5, tolerance_pct: float = 3.0):
        self.window = window
        self.tolerance_pct = tolerance_pct / 100
        self.sr_detector = SupportResistanceDetector(window=window)
    
    def _get_recent_pivots(self, df: pd.DataFrame, n_pivots: int = 5, 
                           price_col: str = 'Close') -> List[Dict]:
        """
        Get the N most recent pivot points with their type (peak/valley).
        
        Returns list of dicts: [{'index': int, 'price': float, 'type': 'peak'|'valley'}, ...]
        Sorted by index (chronological order).
        """
        prices = df[price_col].values
        
        peak_indices, valley_indices = self.sr_detector.find_pivot_points(df, price_col)
        
        # Combine peaks and valleys with type labels
        pivots = []
        for idx in peak_indices:
            pivots.append({'index': idx, 'price': prices[idx], 'type': 'peak'})
        for idx in valley_indices:
            pivots.append({'index': idx, 'price': prices[idx], 'type': 'valley'})
        
        # Sort by index (chronological)
        pivots = sorted(pivots, key=lambda x: x['index'])
        
        # Return the last n_pivots
        return pivots[-n_pivots:] if len(pivots) >= n_pivots else pivots
    
    def _prices_match(self, p1: float, p2: float) -> bool:
        """
        Check if two prices are approximately equal within tolerance.
        
        Two prices match if: |p1 - p2| / avg(p1, p2) <= tolerance_pct
        """
        avg_price = (p1 + p2) / 2
        return abs(p1 - p2) / avg_price <= self.tolerance_pct
    
    def detect_head_and_shoulders(self, pivots: List[Dict]) -> Optional[str]:
        """
        Detect Head and Shoulders pattern.
        
        Pattern structure (5 pivots):
        1. Peak (Left Shoulder)
        2. Valley (Left Armpit)
        3. Higher Peak (Head) - must be higher than both shoulders
        4. Valley (Right Armpit) - roughly same level as left armpit
        5. Lower Peak (Right Shoulder) - roughly same level as left shoulder
        
        For Inverse H&S, the logic is reversed (valleys become peaks).
        """
        if len(pivots) < 5:
            return None
        
        # Get last 5 pivots
        p = pivots[-5:]
        
        # Check for standard Head and Shoulders (bearish)
        # Pattern: peak, valley, higher peak, valley, lower peak
        if (p[0]['type'] == 'peak' and p[1]['type'] == 'valley' and 
            p[2]['type'] == 'peak' and p[3]['type'] == 'valley' and 
            p[4]['type'] == 'peak'):
            
            left_shoulder = p[0]['price']
            left_armpit = p[1]['price']
            head = p[2]['price']
            right_armpit = p[3]['price']
            right_shoulder = p[4]['price']
            
            # Conditions:
            # 1. Head is higher than both shoulders
            # 2. Shoulders are approximately equal
            # 3. Armpits (neckline) are approximately equal
            if (head > left_shoulder and head > right_shoulder and
                self._prices_match(left_shoulder, right_shoulder) and
                self._prices_match(left_armpit, right_armpit)):
                return "Bearish Head & Shoulders"
        
        # Check for Inverse Head and Shoulders (bullish)
        # Pattern: valley, peak, lower valley, peak, higher valley
        if (p[0]['type'] == 'valley' and p[1]['type'] == 'peak' and 
            p[2]['type'] == 'valley' and p[3]['type'] == 'peak' and 
            p[4]['type'] == 'valley'):
            
            left_shoulder = p[0]['price']
            left_armpit = p[1]['price']
            head = p[2]['price']
            right_armpit = p[3]['price']
            right_shoulder = p[4]['price']
            
            # Conditions (inverted):
            # 1. Head is lower than both shoulders
            # 2. Shoulders are approximately equal
            # 3. Armpits (neckline) are approximately equal
            if (head < left_shoulder and head < right_shoulder and
                self._prices_match(left_shoulder, right_shoulder) and
                self._prices_match(left_armpit, right_armpit)):
                return "Bullish Inverse Head & Shoulders"
        
        return None
    
    def detect_double_pattern(self, pivots: List[Dict]) -> Optional[str]:
        """
        Detect Double Top or Double Bottom patterns.
        
        Double Bottom (bullish):
        - Two valleys at approximately the same price level
        - Separated by a peak
        - Last 3 pivots: valley, peak, valley (with valleys matching)
        
        Double Top (bearish):
        - Two peaks at approximately the same price level
        - Separated by a valley
        - Last 3 pivots: peak, valley, peak (with peaks matching)
        """
        if len(pivots) < 3:
            return None
        
        # Get last 3 pivots
        p = pivots[-3:]
        
        # Double Bottom: valley, peak, valley
        if (p[0]['type'] == 'valley' and p[1]['type'] == 'peak' and 
            p[2]['type'] == 'valley'):
            
            if self._prices_match(p[0]['price'], p[2]['price']):
                # Additional check: the peak should be notably higher
                avg_valley = (p[0]['price'] + p[2]['price']) / 2
                if p[1]['price'] > avg_valley * (1 + self.tolerance_pct):
                    return "Bullish Double Bottom"
        
        # Double Top: peak, valley, peak
        if (p[0]['type'] == 'peak' and p[1]['type'] == 'valley' and 
            p[2]['type'] == 'peak'):
            
            if self._prices_match(p[0]['price'], p[2]['price']):
                # Additional check: the valley should be notably lower
                avg_peak = (p[0]['price'] + p[2]['price']) / 2
                if p[1]['price'] < avg_peak * (1 - self.tolerance_pct):
                    return "Bearish Double Top"
        
        return None
    
    def detect_triple_pattern(self, pivots: List[Dict]) -> Optional[str]:
        """
        Detect Triple Top or Triple Bottom patterns.
        
        Triple Bottom (bullish):
        - Three valleys at approximately the same price level
        - Last 5 pivots: valley, peak, valley, peak, valley (valleys matching)
        
        Triple Top (bearish):
        - Three peaks at approximately the same price level
        - Last 5 pivots: peak, valley, peak, valley, peak (peaks matching)
        """
        if len(pivots) < 5:
            return None
        
        p = pivots[-5:]
        
        # Triple Bottom
        if (p[0]['type'] == 'valley' and p[1]['type'] == 'peak' and 
            p[2]['type'] == 'valley' and p[3]['type'] == 'peak' and 
            p[4]['type'] == 'valley'):
            
            v1, v2, v3 = p[0]['price'], p[2]['price'], p[4]['price']
            if self._prices_match(v1, v2) and self._prices_match(v2, v3):
                return "Bullish Triple Bottom"
        
        # Triple Top
        if (p[0]['type'] == 'peak' and p[1]['type'] == 'valley' and 
            p[2]['type'] == 'peak' and p[3]['type'] == 'valley' and 
            p[4]['type'] == 'peak'):
            
            pk1, pk2, pk3 = p[0]['price'], p[2]['price'], p[4]['price']
            if self._prices_match(pk1, pk2) and self._prices_match(pk2, pk3):
                return "Bearish Triple Top"
        
        return None
    
    def detect_pattern(self, df: pd.DataFrame, price_col: str = 'Close') -> Dict:
        """
        Detect all patterns in the DataFrame.
        
        Returns:
        --------
        Dict with:
            'pattern': str or None - the detected pattern name
            'signal': 'bullish', 'bearish', or None
            'pivots': list of recent pivot points used for detection
            'confidence': float 0-1 based on how well the pattern matches
        """
        pivots = self._get_recent_pivots(df, n_pivots=7, price_col=price_col)
        
        if len(pivots) < 3:
            return {'pattern': None, 'signal': None, 'pivots': pivots, 'confidence': 0}
        
        # Try to detect patterns (in order of complexity)
        pattern = None
        
        # Check Head and Shoulders first (needs 5 pivots)
        pattern = self.detect_head_and_shoulders(pivots)
        
        # Check Triple patterns (needs 5 pivots)
        if pattern is None:
            pattern = self.detect_triple_pattern(pivots)
        
        # Check Double patterns (needs 3 pivots)
        if pattern is None:
            pattern = self.detect_double_pattern(pivots)
        
        # Determine signal
        signal = None
        if pattern:
            signal = 'bullish' if 'Bullish' in pattern else 'bearish'
        
        return {
            'pattern': pattern,
            'signal': signal,
            'pivots': pivots,
            'confidence': 0.7 if pattern else 0  # Basic confidence score
        }


def add_support_resistance_to_df(df: pd.DataFrame, window: int = 5, 
                                  tolerance_pct: float = 1.5) -> pd.DataFrame:
    """
    Convenience function to add support/resistance indicators to a DataFrame.
    
    Adds columns:
    - 'is_pivot_high': Boolean, True if this candle is a local maximum
    - 'is_pivot_low': Boolean, True if this candle is a local minimum
    - 'nearest_support': The nearest support level below current price
    - 'nearest_resistance': The nearest resistance level above current price
    """
    df = df.copy()
    detector = SupportResistanceDetector(window=window, tolerance_pct=tolerance_pct)
    
    peak_indices, valley_indices = detector.find_pivot_points(df)
    
    # Mark pivot points
    df['is_pivot_high'] = False
    df['is_pivot_low'] = False
    df.loc[df.index[peak_indices], 'is_pivot_high'] = True
    df.loc[df.index[valley_indices], 'is_pivot_low'] = True
    
    # Get nearest levels for the last row
    levels = detector.get_nearest_levels(df)
    
    # Add nearest support/resistance (scalar value for the current state)
    df['nearest_support'] = levels['support_levels'][0]['level'] if levels['support_levels'] else np.nan
    df['nearest_resistance'] = levels['resistance_levels'][0]['level'] if levels['resistance_levels'] else np.nan
    
    return df


# ============================================================================
# SAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create a dummy DataFrame with OHLCV data
    np.random.seed(42)
    n = 100
    
    # Generate a random walk for close prices
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    
    # Generate OHLC from close
    df = pd.DataFrame({
        'Date': pd.date_range(start='2025-01-01', periods=n, freq='D'),
        'Open': close + np.random.randn(n) * 0.5,
        'High': close + abs(np.random.randn(n)) * 1.5,
        'Low': close - abs(np.random.randn(n)) * 1.5,
        'Close': close,
        'Volume': np.random.randint(100000, 1000000, n)
    })
    
    print("=" * 60)
    print("SUPPORT & RESISTANCE DETECTION")
    print("=" * 60)
    
    # Initialize detector
    sr_detector = SupportResistanceDetector(window=5, tolerance_pct=1.5, min_touches=2)
    
    # Get all zones
    support_zones, resistance_zones = sr_detector.get_zones(df)
    print(f"\nFound {len(support_zones)} support zones and {len(resistance_zones)} resistance zones")
    
    # Get nearest levels
    levels = sr_detector.get_nearest_levels(df, n_levels=3)
    print(f"\nCurrent Price: {levels['current_price']:.2f}")
    print("\nNearest Support Levels:")
    for s in levels['support_levels']:
        print(f"  - {s['level']:.2f} (touches: {s['touches']}, strength: {s['strength']:.2f})")
    print("\nNearest Resistance Levels:")
    for r in levels['resistance_levels']:
        print(f"  - {r['level']:.2f} (touches: {r['touches']}, strength: {r['strength']:.2f})")
    
    print("\n" + "=" * 60)
    print("PATTERN RECOGNITION")
    print("=" * 60)
    
    # Initialize pattern recognizer
    pattern_detector = PatternRecognizer(window=5, tolerance_pct=3.0)
    
    # Detect patterns
    result = pattern_detector.detect_pattern(df)
    print(f"\nDetected Pattern: {result['pattern']}")
    print(f"Signal: {result['signal']}")
    print(f"Number of pivots analyzed: {len(result['pivots'])}")
    
    if result['pivots']:
        print("\nRecent Pivot Points:")
        for p in result['pivots']:
            print(f"  - Index {p['index']}: {p['type'].upper()} at {p['price']:.2f}")
    
    print("\n" + "=" * 60)
    print("ADDING INDICATORS TO DATAFRAME")
    print("=" * 60)
    
    # Add S/R to dataframe
    df_enhanced = add_support_resistance_to_df(df)
    print(f"\nPivot Highs found: {df_enhanced['is_pivot_high'].sum()}")
    print(f"Pivot Lows found: {df_enhanced['is_pivot_low'].sum()}")
    print(f"Nearest Support: {df_enhanced['nearest_support'].iloc[-1]:.2f}")
    print(f"Nearest Resistance: {df_enhanced['nearest_resistance'].iloc[-1]:.2f}")
