"""
Stock Price Forecasting using AI Models
Supports: Amazon Chronos-2 and Datadog Toto
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Global variables to cache the pipelines
_chronos_pipeline = None
_toto_forecaster = None
_toto_model = None

# Legacy alias for backward compatibility
_pipeline = None
_model_loaded = False


# ============================================================================
# MODEL AVAILABILITY CHECKS
# ============================================================================

def is_chronos_available():
    """Check if Chronos-2 model is available."""
    try:
        from chronos import Chronos2Pipeline
        return True
    except ImportError:
        return False


def is_toto_available():
    """Check if Toto model is available."""
    try:
        # First ensure packaging.requirements is available
        import packaging.requirements
        # Then try to import Toto
        from toto.model.toto import Toto
        return True
    except Exception as e:
        # Catch all exceptions (ImportError, RuntimeError from dependency conflicts, etc.)
        print(f"Toto not available: {e}")
        return False


def get_available_models():
    """Get list of available forecasting models."""
    models = []
    if is_chronos_available():
        models.append(("Chronos-2", "Amazon Chronos-2 - Time series foundation model (Context: 8192, Prediction: 1024)"))
    if is_toto_available():
        models.append(("Toto", "Datadog Toto - Open time series model (Context: 4096, Prediction: 1024)"))
    return models


def is_model_available(model_name: str = None):
    """Check if any/specific forecasting model is available."""
    if model_name == "Chronos-2":
        return is_chronos_available()
    elif model_name == "Toto":
        return is_toto_available()
    else:
        # Legacy: check if any model is available (default to Chronos-2)
        return is_chronos_available() or is_toto_available()


# ============================================================================
# CHRONOS-2 MODEL LOADING
# ============================================================================

def get_pipeline(device_map="cpu"):
    """
    Load and cache the Chronos-2 pipeline.
    Uses CPU by default for broader compatibility.
    Set device_map="cuda" for GPU acceleration.
    """
    global _pipeline, _model_loaded
    
    if _pipeline is not None:
        return _pipeline
    
    try:
        from chronos import Chronos2Pipeline
        
        # Try to detect CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                device_map = "cuda"
                print("CUDA available - using GPU for forecasting")
            else:
                device_map = "cpu"
                print("CUDA not available - using CPU for forecasting")
        except ImportError:
            device_map = "cpu"
        
        print(f"Loading Chronos-2 model on {device_map}...")
        _pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",  # Using smaller model for faster loading
            device_map=device_map
        )
        _model_loaded = True
        print("Chronos-2 model loaded successfully!")
        return _pipeline
    
    except ImportError as e:
        print(f"Chronos not installed: {e}")
        print("Install with: pip install 'chronos-forecasting>=2.0'")
        return None
    except Exception as e:
        print(f"Error loading Chronos model: {e}")
        return None


# ============================================================================
# TOTO MODEL LOADING
# ============================================================================

def get_toto_forecaster():
    """
    Load and cache the Toto forecaster.
    """
    global _toto_forecaster, _toto_model
    
    if _toto_forecaster is not None:
        return _toto_forecaster, _toto_model
    
    try:
        import torch
        from toto.model.toto import Toto
        from toto.inference.forecaster import TotoForecaster
        
        # Detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Toto model on {device}...")
        
        # Load pre-trained Toto model
        _toto_model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(device)
        
        # Optional: compile model for enhanced speed (only on CUDA with PyTorch 2.0+)
        if device == 'cuda':
            try:
                _toto_model.compile()
                print("Toto model compiled for enhanced speed")
            except Exception as e:
                print(f"Could not compile Toto model (not critical): {e}")
        
        _toto_forecaster = TotoForecaster(_toto_model.model)
        print("Toto model loaded successfully!")
        return _toto_forecaster, _toto_model
    
    except ImportError as e:
        print(f"Toto not installed: {e}")
        print("Install with: pip install toto-ts")
        return None, None
    except Exception as e:
        print(f"Error loading Toto model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# DATA PREPARATION
# ============================================================================


def prepare_stock_data_for_forecast(chart_data: pd.DataFrame, max_context_length: int = 2048) -> pd.DataFrame:
    """
    Prepare stock chart data for forecasting.
    
    Args:
        chart_data: DataFrame with columns like 'Date', 'Close', 'Open', 'High', 'Low', 'Volume'
        max_context_length: Maximum number of historical data points to use (Chronos-2 supports up to 8192)
    
    Returns:
        DataFrame formatted for Chronos-2 with 'timestamp', 'target', and 'id' columns
    """
    if chart_data.empty:
        return pd.DataFrame()
    
    # Create a clean dataframe for forecasting
    df = pd.DataFrame()
    
    # Handle date column
    if 'Date' in chart_data.columns:
        dates = pd.to_datetime(chart_data['Date'])
    elif chart_data.index.name == 'Date' or isinstance(chart_data.index, pd.DatetimeIndex):
        dates = pd.to_datetime(chart_data.index)
    else:
        # Create date range if no date column
        dates = pd.date_range(end=datetime.now(), periods=len(chart_data), freq='D')
    
    # Use Close price as the target
    prices = chart_data['Close'].values
    
    # Create temporary df for resampling
    temp_df = pd.DataFrame({'target': prices}, index=dates)
    
    # Sort by index
    temp_df = temp_df.sort_index()
    
    # Remove any NaN values first
    temp_df = temp_df.dropna()
    
    # Resample to daily frequency, forward-fill missing values (weekends/holidays)
    temp_df = temp_df.resample('D').ffill()
    
    # Drop any remaining NaN
    temp_df = temp_df.dropna()
    
    # Limit to max_context_length (use most recent data)
    if len(temp_df) > max_context_length:
        temp_df = temp_df.tail(max_context_length)
        print(f"Using last {max_context_length} data points for context (out of {len(temp_df)} available)")
    else:
        print(f"Using all {len(temp_df)} available data points for context")
    
    # Create final dataframe with required columns
    df = pd.DataFrame({
        'timestamp': temp_df.index,
        'target': temp_df['target'].values,
        'id': 'stock'
    })
    
    return df


# ============================================================================
# TOTO FORECASTING HELPER
# ============================================================================

def _forecast_with_toto(context_df: pd.DataFrame, prediction_length: int, last_date, last_price: float) -> dict:
    """
    Internal function to forecast using Toto model.
    """
    import torch
    from toto.data.util.dataset import MaskedTimeseries
    
    forecaster, model = get_toto_forecaster()
    
    if forecaster is None:
        return {
            'success': False,
            'error': 'Failed to load Toto model.',
            'forecast': None
        }
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Prepare input for Toto
        # Toto expects shape: (num_variables, num_timesteps)
        input_series = torch.tensor(context_df['target'].values, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Calculate timestamps in seconds (from first date)
        timestamps = context_df['timestamp']
        first_ts = timestamps.iloc[0]
        timestamp_seconds = torch.tensor(
            [(ts - first_ts).total_seconds() for ts in timestamps],
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        # Daily interval = 86400 seconds
        time_interval_seconds = torch.tensor([86400.0], dtype=torch.float32).to(device)
        
        # Create MaskedTimeseries input
        inputs = MaskedTimeseries(
            series=input_series,
            padding_mask=torch.ones_like(input_series, dtype=torch.bool),
            id_mask=torch.zeros_like(input_series),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )
        
        # Generate forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            forecast = forecaster.forecast(
                inputs,
                prediction_length=prediction_length,
                num_samples=256,
                samples_per_batch=256,
            )
        
        # Extract results
        median = forecast.median.cpu().numpy().flatten().tolist()
        lower_90 = forecast.quantile(0.1).cpu().numpy().flatten().tolist()
        upper_90 = forecast.quantile(0.9).cpu().numpy().flatten().tolist()
        lower_50 = forecast.quantile(0.25).cpu().numpy().flatten().tolist()
        upper_50 = forecast.quantile(0.75).cpu().numpy().flatten().tolist()
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=prediction_length,
            freq='D'
        )
        
        # Convert dates to string format
        forecast_dates = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        # Convert last_date to string
        if hasattr(last_date, 'strftime'):
            last_date_str = last_date.strftime('%Y-%m-%d')
        else:
            last_date_str = str(last_date)[:10]
        
        print(f"Toto forecast generated: {len(median)} predictions")
        
        return {
            'success': True,
            'error': None,
            'model': 'Toto',
            'dates': forecast_dates,
            'median': median,
            'lower_90': lower_90,
            'upper_90': upper_90,
            'lower_50': lower_50,
            'upper_50': upper_50,
            'last_actual_price': last_price,
            'last_actual_date': last_date_str,
            'prediction_length': prediction_length,
            'context_length': len(context_df)
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Toto forecasting error: {error_details}")
        return {
            'success': False,
            'error': f'Toto forecasting error: {str(e)}',
            'forecast': None
        }


# ============================================================================
# MAIN FORECAST FUNCTION
# ============================================================================

def forecast_stock_price(
    chart_data: pd.DataFrame,
    prediction_length: int = 30,
    context_length: int = 2048,
    model_name: str = "Chronos-2",
    quantile_levels: list = [0.1, 0.25, 0.5, 0.75, 0.9]
) -> dict:
    """
    Forecast stock prices using AI models (Chronos-2 or Toto).
    
    Args:
        chart_data: Historical stock data with 'Date' and 'Close' columns
        prediction_length: Number of days to forecast
        context_length: Number of historical data points to use
        model_name: "Chronos-2" or "Toto"
        quantile_levels: Quantile levels for probabilistic forecast
    
    Returns:
        Dictionary with forecast results or error message
    """
    # Enforce model limits
    if model_name == "Toto":
        prediction_length = min(prediction_length, 1024)
        context_length = min(context_length, 4096)
    else:  # Chronos-2
        prediction_length = min(prediction_length, 1024)
        context_length = min(context_length, 8192)
    
    # Check if model is available
    if not is_model_available(model_name):
        if model_name == "Toto":
            return {
                'success': False,
                'error': 'Toto not installed. Run: pip install toto-ts',
                'forecast': None
            }
        else:
            return {
                'success': False,
                'error': 'Chronos-2 not installed. Run: pip install "chronos-forecasting>=2.0"',
                'forecast': None
            }
    
    try:
        # Prepare data with specified context length
        context_df = prepare_stock_data_for_forecast(chart_data, max_context_length=context_length)
        
        if context_df.empty or len(context_df) < 10:
            return {
                'success': False,
                'error': 'Insufficient historical data for forecasting. Need at least 10 data points.',
                'forecast': None
            }
        
        # Get the last date and price from historical data
        last_date = context_df['timestamp'].max()
        last_price = float(context_df['target'].iloc[-1])
        
        # ================================================================
        # TOTO MODEL FORECASTING
        # ================================================================
        if model_name == "Toto":
            return _forecast_with_toto(context_df, prediction_length, last_date, last_price)
        
        # ================================================================
        # CHRONOS-2 MODEL FORECASTING (default)
        # ================================================================
        # Get the pipeline
        pipeline = get_pipeline()
        
        if pipeline is None:
            return {
                'success': False,
                'error': 'Failed to load Chronos-2 model.',
                'forecast': None
            }
        
        # Get the last date and price from historical data
        last_date = context_df['timestamp'].max()
        last_price = float(context_df['target'].iloc[-1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Use predict_df with the correct API from Chronos-2
            pred_df = pipeline.predict_df(
                context_df,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )
        
        # Debug: print columns to see the actual structure
        print(f"Prediction DataFrame columns: {pred_df.columns.tolist()}")
        print(f"Prediction DataFrame head:\n{pred_df.head()}")
        
        # Find the quantile columns dynamically
        cols = pred_df.columns.tolist()
        
        # Try different possible column naming conventions
        if 'target_0.5' in cols:
            median = pred_df['target_0.5'].tolist()
            lower_90 = pred_df['target_0.1'].tolist()
            upper_90 = pred_df['target_0.9'].tolist()
            lower_50 = pred_df['target_0.25'].tolist()
            upper_50 = pred_df['target_0.75'].tolist()
        elif '0.5' in cols:
            median = pred_df['0.5'].tolist()
            lower_90 = pred_df['0.1'].tolist()
            upper_90 = pred_df['0.9'].tolist()
            lower_50 = pred_df['0.25'].tolist()
            upper_50 = pred_df['0.75'].tolist()
            # Use dates from prediction dataframe
            future_dates = pred_df['timestamp'].tolist()
        elif 'quantile_0.5' in cols:
            median = pred_df['quantile_0.5'].tolist()
            lower_90 = pred_df['quantile_0.1'].tolist()
            upper_90 = pred_df['quantile_0.9'].tolist()
            lower_50 = pred_df['quantile_0.25'].tolist()
            upper_50 = pred_df['quantile_0.75'].tolist()
        elif 'mean' in cols:
            # If using mean/std output format
            median = pred_df['mean'].tolist()
            if 'std' in cols:
                std = pred_df['std'].values
                mean_vals = pred_df['mean'].values
                lower_90 = (mean_vals - 1.645 * std).tolist()
                upper_90 = (mean_vals + 1.645 * std).tolist()
                lower_50 = (mean_vals - 0.675 * std).tolist()
                upper_50 = (mean_vals + 0.675 * std).tolist()
            else:
                lower_90 = median
                upper_90 = median
                lower_50 = median
                upper_50 = median
        else:
            # Try to find any quantile-like columns
            quantile_cols = [c for c in cols if any(q in str(c) for q in ['0.1', '0.5', '0.9', '0.25', '0.75'])]
            if quantile_cols:
                print(f"Found quantile columns: {quantile_cols}")
                # Use the first numeric column as median
                numeric_cols = [c for c in cols if c not in ['id', 'timestamp', 'ds', 'unique_id']]
                if numeric_cols:
                    median = pred_df[numeric_cols[0]].tolist()
                    lower_90 = median
                    upper_90 = median
                    lower_50 = median
                    upper_50 = median
            else:
                # Just use any numeric column
                numeric_cols = [c for c in cols if c not in ['id', 'timestamp', 'ds', 'unique_id']]
                if numeric_cols:
                    median = pred_df[numeric_cols[0]].tolist()
                    lower_90 = median
                    upper_90 = median
                    lower_50 = median
                    upper_50 = median
                else:
                    raise ValueError(f"Could not find forecast columns. Available columns: {cols}")
        
        # Generate future dates for the forecast (if not already set)
        if 'future_dates' not in dir() or future_dates is None:
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=prediction_length,
                freq='D'
            ).tolist()
        
        # Convert dates to string format for Plotly compatibility
        if hasattr(future_dates, 'tolist'):
            future_dates = future_dates.tolist()
        
        # Convert Timestamp objects to datetime strings
        forecast_dates = []
        for d in future_dates:
            if hasattr(d, 'strftime'):
                forecast_dates.append(d.strftime('%Y-%m-%d'))
            else:
                forecast_dates.append(str(d))
        
        # Also convert last_actual_date to string
        if hasattr(last_date, 'strftime'):
            last_date_str = last_date.strftime('%Y-%m-%d')
        else:
            last_date_str = str(last_date)
        
        # Extract forecast values
        forecast_result = {
            'success': True,
            'error': None,
            'model': 'Chronos-2',
            'dates': forecast_dates,
            'median': median,
            'lower_90': lower_90,
            'upper_90': upper_90,
            'lower_50': lower_50,
            'upper_50': upper_50,
            'last_actual_price': last_price,
            'last_actual_date': last_date_str,
            'prediction_length': prediction_length,
            'context_length': len(context_df)
        }
        
        return forecast_result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Forecasting error details: {error_details}")
        return {
            'success': False,
            'error': f'Forecasting error: {str(e)}',
            'forecast': None
        }


def get_forecast_summary(forecast_result: dict) -> dict:
    """
    Generate a summary of the forecast results.
    
    Args:
        forecast_result: Output from forecast_stock_price()
    
    Returns:
        Dictionary with forecast summary statistics
    """
    if not forecast_result.get('success'):
        return None
    
    median = forecast_result.get('median', [])
    last_price = forecast_result.get('last_actual_price', 0)
    
    if not median or not last_price:
        return None
    
    # Calculate statistics
    forecast_end = median[-1]
    forecast_high = max(median)
    forecast_low = min(median)
    
    price_change = forecast_end - last_price
    price_change_pct = (price_change / last_price) * 100 if last_price > 0 else 0
    
    # Determine trend
    if price_change_pct > 5:
        trend = "Strong Bullish"
        trend_emoji = "🚀"
    elif price_change_pct > 2:
        trend = "Bullish"
        trend_emoji = "📈"
    elif price_change_pct > -2:
        trend = "Neutral"
        trend_emoji = "➡️"
    elif price_change_pct > -5:
        trend = "Bearish"
        trend_emoji = "📉"
    else:
        trend = "Strong Bearish"
        trend_emoji = "⚠️"
    
    return {
        'current_price': last_price,
        'forecast_end_price': forecast_end,
        'forecast_high': forecast_high,
        'forecast_low': forecast_low,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'trend': trend,
        'trend_emoji': trend_emoji,
        'days_ahead': len(median),
        'model': forecast_result.get('model', 'Unknown')
    }
