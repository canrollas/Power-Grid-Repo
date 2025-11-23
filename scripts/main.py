#!/usr/bin/env python3
"""
Data Reading Script - Grid Consumption Data

Loads the dataset, displays information, and plots consumption for 3 random clients.

File format:
- Semicolon separated (';')
- First column: 'yyyy-mm-dd hh:mm:ss' datetime string
- Other columns: float values (kW)
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_categorization import analyze_client_categories
from lstm_training import run_lstm_training

def load_consumption_data(data_path="data/raw/LD2011_2014.txt", console=None):
    """
    Loads electricity consumption data.

    Args:
        data_path: Path to the data file.
        console: Rich Console object for colored output.

    Returns:
        DataFrame: Loaded data with datetime index and float columns.
    """
    if console is None:
        console = Console()
    
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    console.print("   [yellow]Reading file (this may take a few minutes)...[/yellow]")
    
    # Read CSV as semicolon-separated
    # Use 'c' engine for better performance with large files
    # Note: Data uses comma as decimal separator, so we read as string first
    console.print("   [cyan]Reading data (this may take some time)...[/cyan]")
    try:
        # Try with date_format (pandas >= 2.0.0)
        # Read as string first to handle comma decimal separator
        df = pd.read_csv(
            data_path,
            sep=';',
            header=0,
            index_col=0,
            parse_dates=[0],
            date_format='%Y-%m-%d %H:%M:%S',
            dtype=str,
            engine='c',
            low_memory=False
        )
    except (TypeError, ValueError):
        # For pandas <= 2.0.0 or if date_format fails
        console.print("   [yellow]Trying alternative reading method...[/yellow]")
        df = pd.read_csv(
            data_path,
            sep=';',
            header=0,
            index_col=0,
            parse_dates=[0],
            dtype=str,
            engine='c',
            low_memory=False
        )
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Convert comma decimal separator to dot and convert to float
    console.print("   [cyan]Converting data types (replacing comma with dot for decimal numbers)...[/cyan]")
    for col in df.columns:
        try:
            # Replace comma with dot for decimal separator (European format)
            df[col] = df[col].str.replace(',', '.', regex=False)
            # Convert to float
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            console.print(f"   [yellow]Warning: Column {col} could not be converted to float: {e}[/yellow]")

    console.print("   [green]Data loaded successfully![/green]")
    # The index is now datetime, columns are str (client IDs), values are floats (kW)
    return df

def trim_leading_zeros(data, threshold=0.01):
    """
    Removes leading zeros (or values very close to zero) from the beginning of data.
    
    Args:
        data: Numpy array or pandas Series
        threshold: Minimum value to consider as active (default: 0.01)
    
    Returns:
        Trimmed data array, or None if data is completely empty/zero
    """
    data = np.array(data)
    
    # Find indices where data is greater than threshold
    non_zero_indices = np.where(data > threshold)[0]
    
    if len(non_zero_indices) == 0:
        return None  # Data is completely empty or all zeros
    
    first_active_idx = non_zero_indices[0]
    
    # Return data from first active index onwards
    return data[first_active_idx:]

def apply_auto_trimming(df, console=None, min_length=500):
    """
    Applies auto-trimming to remove leading zeros from all client columns.
    
    Args:
        df: DataFrame with datetime index and client columns
        console: Rich Console object for colored output
        min_length: Minimum length after trimming to keep the client (default: 500)
    
    Returns:
        Trimmed DataFrame with same structure
    """
    if console is None:
        console = Console()
    
    console.print("\n[bold]4. Applying Auto-Trimming (Cold Start Problem Fix):[/bold]")
    console.print(f"   [dim]Processing {len(df.columns)} clients...[/dim]")
    
    trimmed_df = df.copy()
    skipped_clients = []
    trimmed_info = {}
    
    # Use progress bar to show processing of all clients
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Trimming clients...", total=len(df.columns))
        
        for col in df.columns:
            raw_data = df[col].values
            
            # Find first active index
            clean_data = trim_leading_zeros(raw_data)
            
            if clean_data is None or len(clean_data) < min_length:
                # Skip this client - data too short after trimming
                trimmed_df = trimmed_df.drop(columns=[col])
                skipped_clients.append(col)
                progress.update(task, advance=1)
                continue
            
            first_active_idx = len(raw_data) - len(clean_data)
            trimmed_info[col] = {
                'original_length': len(raw_data),
                'trimmed_length': len(clean_data),
                'trimmed_points': first_active_idx,
                'first_active_date': df.index[first_active_idx]
            }
            
            # Update the column with trimmed data (pad with NaN at the beginning)
            trimmed_values = np.full(len(raw_data), np.nan)
            trimmed_values[first_active_idx:] = clean_data
            trimmed_df[col] = trimmed_values
            
            progress.update(task, advance=1)
    
    # Drop rows where all values are NaN (before any client started)
    trimmed_df = trimmed_df.dropna(how='all')
    
    console.print(f"   [green]Successfully trimmed {len(trimmed_info)} clients[/green]")
    if skipped_clients:
        console.print(f"   [yellow]Skipped {len(skipped_clients)} clients (too short after trimming)[/yellow]")
    console.print(f"   [dim]Total processed: {len(df.columns)} clients[/dim]")
    
    return trimmed_df, trimmed_info, skipped_clients

def main():
    """
    Loads the dataset, displays information, and plots consumption for 3 random clients.
    """
    console = Console()
    
    # Print header with panel
    title = Text("DATA READING - Grid Consumption Data", style="bold blue")
    console.print(Panel(title, border_style="blue", expand=False))
    
    data_path = "data/raw/LD2011_2014.txt"
    console.print(f"\n[bold]1. Loading data file:[/bold] [cyan]{data_path}[/cyan]")

    try:
        df = load_consumption_data(data_path, console)
        console.print("\n[bold]2. Dataset Info:[/bold]")
        console.print(f"   [dim]- Number of datetime entries (rows):[/dim] [green]{df.shape[0]:,}[/green]")
        console.print(f"   [dim]- Number of clients (columns):[/dim] [green]{df.shape[1]}[/green]")
        console.print(f"   [dim]- Time range:[/dim] [green]{df.index.min()} - {df.index.max()}[/green]")
        client_cols_str = ', '.join(map(str, df.columns[:min(5, len(df.columns))]))
        if len(df.columns) > 5:
            client_cols_str += '...'
        console.print(f"   [dim]- Client columns:[/dim] [green]{client_cols_str}[/green]")

        # Client Categorization Analysis (Before Trimming)
        console.print("\n" + "="*60)
        stats_before, client_metrics_before = analyze_client_categories(df, console, title_suffix="(Before Trimming)")
        console.print("="*60)

        # Select 3 unique random clients
        client_ids = list(df.columns)
        if len(client_ids) < 3:
            console.print("[red]   Error: Not enough clients in the dataset to select 3.[/red]")
            sys.exit(1)
        random_clients = random.sample(client_ids, 3)
        
        console.print(f"\n[bold]3. Plotting consumption for 3 random clients:[/bold] [cyan]{', '.join(map(str, random_clients))}[/cyan]")

        plt.figure(figsize=(14, 6))
        for cid in random_clients:
            plt.plot(df.index, df[cid], label=f"Client {cid}", linewidth=1)
        plt.title("Electricity Consumption of 3 Random Clients (Original Data)")
        plt.xlabel("Datetime")
        plt.ylabel("Consumption (kW)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Cold Start Problem Explanation and Auto-Trimming
        console.print("\n[bold]4. Cold Start Problem & Auto-Trimming:[/bold]")
        console.print("   [dim]Removing leading zeros to prevent model learning 'zero consumption is normal'[/dim]")
        
        # Apply auto-trimming
        trimmed_df, trimmed_info, skipped_clients = apply_auto_trimming(df, console)
        
        # Show trimming info for the 3 random clients (simplified)
        console.print("\n[bold]5. Trimming Summary:[/bold]")
        console.print(f"   [green]Successfully processed {len(trimmed_info)} clients[/green]")
        if skipped_clients:
            console.print(f"   [yellow]Skipped {len(skipped_clients)} clients (too short)[/yellow]")
        
        # Plot trimmed data for the same clients (if they still exist)
        available_clients = [cid for cid in random_clients if cid in trimmed_df.columns]
        if available_clients:
            console.print(f"\n[bold]6. Plotting trimmed consumption data:[/bold] [cyan]{', '.join(map(str, available_clients))}[/cyan]")
            
            plt.figure(figsize=(14, 6))
            for cid in available_clients:
                # Only plot non-NaN values
                mask = ~trimmed_df[cid].isna()
                if mask.sum() > 0:
                    plt.plot(trimmed_df.index[mask], trimmed_df[cid][mask], 
                            label=f"Client {cid} (trimmed)", linewidth=1)
            plt.title("Electricity Consumption - After Auto-Trimming (Cold Start Fixed)")
            plt.xlabel("Datetime")
            plt.ylabel("Consumption (kW)")
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Client Categorization Analysis (After Trimming)
        # 
        # IMPORTANT NOTE: Why do categorization results change after trimming?
        #
        # 1. Why "High Variance" Decreases:
        #    Before trimming, a client's data looked like: [0, 0, 0, 0, ..., 0, 500, 520, 510]
        #    Statistically, thousands of zeros followed by sudden values like 500 causes
        #    the standard deviation (variance) to become enormous. This is why the code
        #    mistakenly classified these clients as "High Variance" (Irregular).
        #
        #    After removing leading zeros, the data becomes: [500, 520, 510]
        #    Now the variance drops significantly, and these clients shift to "Stable"
        #    (Regular) or "Normal" category. This is why the percentage drops from ~17% to ~3%.
        #
        # 2. Why "Sparse Data" Increases:
        #    This is the critical point. Trimming only removes the leading block of zeros
        #    (Cold Start). However, it does NOT remove zeros within the data (interruptions).
        #
        #    Previously, due to the massive Cold Start block, small interruptions within
        #    the data were statistically masked. After cleaning, profiles like this emerge:
        #    [100, 0, 0, 100, 0, 100] (Machine ran, stopped, ran again).
        #
        #    This profile appeared as "High Variance" when Cold Start was present.
        #    After removing the leading excess, it becomes clear that this client's data
        #    is actually sparse (intermittent). The code can now correctly diagnose:
        #    "This client does not consume regularly, experiences constant interruptions."
        #
        console.print("\n" + "="*60)
        stats_after, client_metrics_after = analyze_client_categories(trimmed_df, console, title_suffix="(After Trimming)")
        console.print("="*60)
        
        # Update dataset info
        console.print("\n[bold]7. Updated Dataset Info (After Trimming):[/bold]")
        console.print(f"   [dim]- Number of datetime entries (rows):[/dim] [green]{trimmed_df.shape[0]:,}[/green]")
        console.print(f"   [dim]- Number of clients (columns):[/dim] [green]{trimmed_df.shape[1]}[/green]")
        if trimmed_df.shape[0] > 0:
            console.print(f"   [dim]- Time range:[/dim] [green]{trimmed_df.index.min()} - {trimmed_df.index.max()}[/green]")
        
        # LSTM Training with Optuna
        console.print("\n" + "="*60)
        lstm_results, failed_clients = run_lstm_training(
            trimmed_df,
            console=console,
            sequence_length=24,
            forecast_horizon=1,
            n_trials=20,
            max_clients=None  # Set to a number to limit clients for testing
        )
        console.print("="*60)
        
        console.print("\n[green][bold]Analysis completed successfully![/bold][/green]")
        console.print("[dim]The dataset has been transformed to remove Cold Start problems.[/dim]")

    except FileNotFoundError:
        console.print(f"[red]   Error: Data file not found: {data_path}[/red]")
        sys.exit(1)
    except ValueError as ve:
        console.print(f"[red]   Error: {ve}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]   Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
