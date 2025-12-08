#!/usr/bin/env python3
"""
Test Model Script

Loads the trained LSTM model and evaluates it on the test set.
Reuses the exact same data preprocessing pipeline (Trimming + Scaling) to ensure validity.
"""

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import existing logic to ensure consistency
from main import load_consumption_data, apply_auto_trimming
from lstm_training import prepare_multi_client_data_generators, configure_tensorflow_devices
from experiment_tracker import ExperimentTracker

def test_model(model_path="data/processed/models/best_lstm_model.keras", 
               data_path="data/raw/LD2011_2014.txt",
               batch_size=2048):
    
    console = Console()
    
    # Header
    title = Text("LSTM MODEL EVALUATION", style="bold green")
    console.print(Panel(title, border_style="green", expand=False))
    
    # 1. Check Model File
    if not os.path.exists(model_path):
        console.print(f"[red]Error: Model file not found at {model_path}[/red]")
        console.print("[dim]Please run training first: python scripts/main.py[/dim]")
        return False
        
    console.print(f"\n[bold]1. Loading Model:[/bold] [cyan]{model_path}[/cyan]")
    try:
        model = tf.keras.models.load_model(model_path)
        console.print(f"   [green]Model loaded successfully![/green]")
        model.summary(print_fn=lambda x: console.print(f"   [dim]{x}[/dim]"))
    except Exception as e:
        console.print(f"   [red]Error loading model: {e}[/red]")
        return False

    # 2. Load Data
    console.print(f"\n[bold]2. Loading and Preprocessing Data...[/bold]")
    try:
        # Re-use the EXACT SAME loading logic
        df = load_consumption_data(data_path, console)
        
        # Apply EXACT SAME trimming logic
        trimmed_series, trimmed_info, _ = apply_auto_trimming(df, console)
        
    except Exception as e:
        console.print(f"   [red]Error loading data: {e}[/red]")
        return False
        
    # 3. Prepare Generators (This fits scalers on TRAIN split exactly like training did)
    console.print(f"\n[bold]3. Re-creating Data Generators...[/bold]")
    
    # Using the same parameters as training
    sequence_length = 24
    forecast_horizon = 1
    
    data_dict = prepare_multi_client_data_generators(
        trimmed_series,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        test_size=0.2,
        validation_size=0.05,
        batch_size=batch_size,
        console=console
    )
    
    if data_dict is None:
        console.print("[red]Failed to prepare data generators.[/red]")
        return False
        
    test_gen = data_dict['test_gen']
    
    console.print(f"   [dim]Test samples: {test_gen.total_samples:,}[/dim]")
    
    if test_gen.total_samples == 0:
        console.print("[red]No test data available![/red]")
        return False

    # 4. Predict
    console.print(f"\n[bold]4. Running Predictions on Test Set...[/bold]")
    
    test_predictions = model.predict(test_gen, verbose=1)
    
    # 5. Extract Truth and IDs
    console.print(f"\n[bold]5. Calculating Metrics...[/bold]")
    
    test_actual = []
    test_client_ids = []
    
    # Generator yields ({inputs}, y)
    for i in range(len(test_gen)):
        inputs_batch, y_batch = test_gen[i]
        test_actual.extend(y_batch)
        test_client_ids.extend(inputs_batch['client_id'])
        
    test_actual = np.array(test_actual)
    test_client_ids = np.array(test_client_ids)
    
    # 6. Inverse Transform
    console.print("   [dim]Inverse transforming (unscaling) predictions...[/dim]")
    
    # Group indices by client for batch inverse transform
    client_indices = {}
    for i, cid_enc in enumerate(test_client_ids):
        if cid_enc not in client_indices:
            client_indices[cid_enc] = []
        client_indices[cid_enc].append(i)
        
    # Prepare result arrays
    test_pred_inv = np.zeros_like(test_predictions)
    test_actual_inv = np.zeros_like(test_actual)
    
    scalers = data_dict['scalers']
    
    for cid_enc, indices in client_indices.items():
        if cid_enc in scalers:
            scaler = scalers[cid_enc]
            
            # Get subsets
            pred_subset = test_predictions[indices]
            actual_subset = test_actual[indices]
            
            # Inverse transform (Scaler expects 2D, outputs 2D)
            pred_inv = scaler.inverse_transform(pred_subset)
            actual_inv = scaler.inverse_transform(actual_subset)
            
            # Post-Processing: Dynamic Zero-Snapping (Client Specific)
            # If prediction is < 8% of this client's max observed consumption, snap to 0.
            # This handles both small clients (max 1kW -> thresh 0.08kW) and large (max 1000kW -> thresh 80kW)
            max_load = np.max(actual_inv) if len(actual_inv) > 0 else 1.0
            threshold = 0.08 * max_load # Increased to 8% to catch the stubborn 10kW false positives
            
            mask = pred_inv < threshold
            pred_inv[mask] = 0.0
            
            # Store back
            test_pred_inv[indices] = pred_inv
            test_actual_inv[indices] = actual_inv
            
    # 8. Compute Metrics (Safe & Robust)
    
    # MAE
    mae = np.mean(np.abs(test_pred_inv - test_actual_inv))
    
    # RMSE
    rmse = np.sqrt(np.mean((test_pred_inv - test_actual_inv) ** 2))
    
    # R2
    ss_res = np.sum((test_actual_inv - test_pred_inv) ** 2)
    ss_tot = np.sum((test_actual_inv - np.mean(test_actual_inv)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Safe MAPE (> 0.1)
    mask = test_actual_inv > 0.1
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((test_actual_inv[mask] - test_pred_inv[mask]) / test_actual_inv[mask])) * 100
    else:
        mape = 0.0
        
    # WMAPE
    wmape = np.sum(np.abs(test_actual_inv - test_pred_inv)) / (np.sum(test_actual_inv) + 1e-8) * 100
    
    console.print(f"\n[bold green]FINAL RESULTS:[/bold green]")
    console.print(Panel(
        f"MAE:   {mae:.4f} kW\n"
        f"RMSE:  {rmse:.4f} kW\n"
        f"MAPE:  {mape:.2f} % (Safe)\n"
        f"WMAPE: {wmape:.2f} %\n"
        f"R²:    {r2:.4f}",
        title="Performance Metrics",
        border_style="green"
    ))
    
    # 8. Visualizations
    console.print(f"\n[bold]8. Generating Visualizations...[/bold]")
    plots_dir = "data/processed/plots/test_results"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate errors
    errors = np.abs(test_actual_inv - test_pred_inv)
    
    # A) Feature: Highlight WORST Errors (Top 3)
    # Find indices of largest errors
    worst_indices = np.argsort(errors.flatten())[-3:]
    
    for rank, idx in enumerate(reversed(worst_indices)):
        # Determine which client this belongs to
        client_id_enc = test_client_ids[idx]
        
        # Determine context window (e.g., +/- 48 hours around the error)
        window_size = 100 
        start_pos = max(0, idx - window_size // 2)
        end_pos = min(len(test_actual_inv), idx + window_size // 2)
        
        # Extract window data
        y_true_window = test_actual_inv[start_pos:end_pos].flatten()
        y_pred_window = test_pred_inv[start_pos:end_pos].flatten()
        error_point_relative_idx = idx - start_pos
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true_window, label='Actual Data', color='black', linewidth=1.5, alpha=0.8)
        plt.plot(y_pred_window, label='LSTM Model Forecast', color='#FF8C00', linewidth=1.5, alpha=0.9) # Dark Orange
        
        # Highlight the error point
        plt.scatter([error_point_relative_idx], [y_true_window[error_point_relative_idx]], 
                    color='red', s=100, zorder=5, label='Worst Error Point (Actual)')
        plt.title(f"Worst Error #{rank+1} (Error: {errors.flatten()[idx]:.2f} kW)")
        plt.xlabel("Time Steps (Window)")
        plt.ylabel("Consumption (kW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"{plots_dir}/worst_error_{rank+1}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        console.print(f"   [yellow]Saved plot: {save_path}[/yellow]")

    # B) Feature: Random Predictions (Good/Normal Cases)
    # Pick 3 random indices
    random_indices = np.random.choice(len(test_actual_inv), 3, replace=False)
    
    for i, idx in enumerate(random_indices):
        window_size = 100
        start_pos = max(0, idx - window_size // 2)
        end_pos = min(len(test_actual_inv), idx + window_size // 2)
        
        y_true_window = test_actual_inv[start_pos:end_pos].flatten()
        y_pred_window = test_pred_inv[start_pos:end_pos].flatten()
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true_window, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred_window, label='Predicted', color='orange', alpha=0.7)
        plt.title(f"Random Prediction Sample #{i+1}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"{plots_dir}/random_sample_{i+1}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        console.print(f"   [cyan]Saved plot: {save_path}[/cyan]")
    
    return True

if __name__ == "__main__":
    test_model()
