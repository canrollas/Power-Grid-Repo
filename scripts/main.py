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
import matplotlib
# Set matplotlib backend to Agg (non-interactive) - works without tkinter
# We save plots to files, so interactive display is not required
matplotlib.use('Agg')
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
from experiment_tracker import ExperimentTracker
from error_handler import ErrorHandler
from report_generator import ReportGenerator

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

def apply_auto_trimming(df, console=None, min_length=500, threshold=0.01):
    """
    Applies client-based trimming: Her client için ilk anlamlı değerden itibaren 
    başlayan ayrı seriler oluşturur. Zaman eksenini fiziksel olarak kısaltır.
    
    Args:
        df: DataFrame with datetime index and client columns
        console: Rich Console object for colored output
        min_length: Minimum length after trimming to keep the client (default: 500)
        threshold: Minimum value to consider as active (default: 0.01)
    
    Returns:
        dict[str, pd.Series]: Her client için trimmed seri (farklı uzunlukta zaman index'i)
        dict: Trimming bilgileri
        list: Atılan client'lar
    """
    if console is None:
        console = Console()
    
    console.print("\n[bold]4. Applying Client-Based Trimming (Cold Start Problem Fix):[/bold]")
    console.print(f"   [dim]Her client için ilk anlamlı değerden itibaren seri oluşturuluyor...[/dim]")
    console.print(f"   [dim]Processing {len(df.columns)} clients...[/dim]")
    
    trimmed_series = {}  # dict[str, pd.Series] - her client'ın kendi zaman serisi
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
            client_series = df[col]  # pandas Series with datetime index
            
            # İlk anlamlı değerin index'ini bul
            active_mask = client_series > threshold
            active_indices = client_series.index[active_mask]
            
            if len(active_indices) == 0:
                # Hiç anlamlı değer yok, client'ı at
                skipped_clients.append(col)
                progress.update(task, advance=1)
                continue
            
            # İlk ve son anlamlı değerlerin index'leri
            t_start = active_indices[0]
            t_end = active_indices[-1]
            
            # Client'ın trimmed serisini oluştur (fiziksel olarak kes)
            s_trimmed = client_series.loc[t_start:t_end].dropna()
            
            if len(s_trimmed) < min_length:
                # Çok kısa, client'ı at
                skipped_clients.append(col)
                progress.update(task, advance=1)
                continue
            
            # Trimmed seriyi dict'e ekle
            trimmed_series[col] = s_trimmed
            
            # Trimming bilgilerini kaydet
            original_length = len(client_series)
            trimmed_length = len(s_trimmed)
            trimmed_points = original_length - trimmed_length
            
            trimmed_info[col] = {
                'original_length': original_length,
                'trimmed_length': trimmed_length,
                'trimmed_points': trimmed_points,
                'first_active_date': t_start,
                'last_active_date': t_end
            }
            
            progress.update(task, advance=1)
    
    console.print(f"   [green]Successfully trimmed {len(trimmed_series)} clients[/green]")
    if skipped_clients:
        console.print(f"   [yellow]Skipped {len(skipped_clients)} clients (too short after trimming)[/yellow]")
    console.print(f"   [dim]Total processed: {len(df.columns)} clients[/dim]")
    
    # Toplam satır sayısını göster (artık her client farklı uzunlukta)
    total_data_points = sum(len(s) for s in trimmed_series.values())
    console.print(f"   [dim]Total data points (sum of all client series): {total_data_points:,}[/dim]")
    console.print(f"   [dim]Average series length: {total_data_points / len(trimmed_series):.0f} points[/dim]")
    
    return trimmed_series, trimmed_info, skipped_clients

def main():
    """
    Loads the dataset, displays information, and plots consumption for 3 random clients.
    """
    console = Console()
    
    # Initialize experiment tracking and error handling
    tracker = ExperimentTracker()
    error_handler = ErrorHandler(tracker=tracker, console=console)
    
    console.print(f"\n[bold]Experiment:[/bold] [cyan]{tracker.experiment_name}[/cyan]")
    console.print(f"[dim]Results will be saved to: {tracker.experiment_dir}[/dim]")
    
    # Print header with panel
    title = Text("DATA READING - Grid Consumption Data", style="bold blue")
    console.print(Panel(title, border_style="blue", expand=False))
    
    data_path = "data/raw/LD2011_2014.txt"
    console.print(f"\n[bold]1. Loading data file:[/bold] [cyan]{data_path}[/cyan]")
    
    tracker.log_step("DATA_LOADING_START", f"Loading data from {data_path}")

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

        # Create output directory for plots if it doesn't exist
        os.makedirs("data/processed/plots", exist_ok=True)
        
        plt.figure(figsize=(14, 6))
        for cid in random_clients:
            plt.plot(df.index, df[cid], label=f"Client {cid}", linewidth=1)
        plt.title("Electricity Consumption of 3 Random Clients (Original Data)")
        plt.xlabel("Datetime")
        plt.ylabel("Consumption (kW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = "data/processed/plots/consumption_before_trimming.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        console.print(f"   [green]Plot saved to: {plot_path}[/green]")
        tracker.add_plot('consumption_before_trimming', plot_path, 'Consumption before trimming')
        plt.close()  # Close figure to free memory
        
        # Cold Start Problem Explanation and Auto-Trimming
        console.print("\n[bold]4. Cold Start Problem & Auto-Trimming:[/bold]")
        console.print("   [dim]Removing leading zeros to prevent model learning 'zero consumption is normal'[/dim]")
        
        # Apply auto-trimming (artık dict[str, pd.Series] döndürüyor)
        trimmed_series, trimmed_info, skipped_clients = apply_auto_trimming(df, console)
        
        # Show trimming info for the 3 random clients (simplified)
        console.print("\n[bold]5. Trimming Summary:[/bold]")
        console.print(f"   [green]Successfully processed {len(trimmed_info)} clients[/green]")
        if skipped_clients:
            console.print(f"   [yellow]Skipped {len(skipped_clients)} clients (too short)[/yellow]")
        
        # Plot trimmed data for the same clients (if they still exist)
        available_clients = [cid for cid in random_clients if cid in trimmed_series]
        if available_clients:
            console.print(f"\n[bold]6. Plotting trimmed consumption data:[/bold] [cyan]{', '.join(map(str, available_clients))}[/cyan]")
            
            plt.figure(figsize=(14, 6))
            for cid in available_clients:
                # Her client'ın serisi zaten trimmed (cold start yok)
                client_series = trimmed_series[cid]
                plt.plot(client_series.index, client_series.values, 
                        label=f"Client {cid} (trimmed)", linewidth=1)
            plt.title("Electricity Consumption - After Auto-Trimming (Cold Start Fixed)")
            plt.xlabel("Datetime")
            plt.ylabel("Consumption (kW)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = "data/processed/plots/consumption_after_trimming.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            console.print(f"   [green]Plot saved to: {plot_path}[/green]")
            tracker.add_plot('consumption_after_trimming', plot_path, 'Consumption after trimming')
            plt.close()  # Close figure to free memory
        
        # Client Categorization Analysis (After Trimming)
        # Categorization için DataFrame'e ihtiyaç var, bu yüzden trimmed_series'den oluşturuyoruz
        # NOT: Bu DataFrame sadece görselleştirme için, asıl işlemler trimmed_series ile yapılıyor
        console.print("\n[bold]6.5. Creating DataFrame for categorization (visualization only)...[/bold]")
        # Tüm client'ların birleşik zaman eksenini bul
        all_timestamps = set()
        for client_series in trimmed_series.values():
            all_timestamps.update(client_series.index)
        all_timestamps = sorted(all_timestamps)
        
        # DataFrame oluştur (NaN ile doldurulacak)
        trimmed_df_for_viz = pd.DataFrame(index=all_timestamps, columns=list(trimmed_series.keys()))
        for client_id, client_series in trimmed_series.items():
            trimmed_df_for_viz.loc[client_series.index, client_id] = client_series.values
        
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
        stats_after, client_metrics_after = analyze_client_categories(trimmed_df_for_viz, console, title_suffix="(After Trimming)")
        console.print("="*60)
        
        # Update dataset info
        console.print("\n[bold]7. Updated Dataset Info (After Trimming):[/bold]")
        total_data_points = sum(len(s) for s in trimmed_series.values())
        avg_length = total_data_points / len(trimmed_series) if len(trimmed_series) > 0 else 0
        console.print(f"   [dim]- Number of clients:[/dim] [green]{len(trimmed_series)}[/green]")
        console.print(f"   [dim]- Total data points (sum of all client series):[/dim] [green]{total_data_points:,}[/green]")
        console.print(f"   [dim]- Average series length:[/dim] [green]{avg_length:.0f} points[/green]")
        if len(trimmed_series) > 0:
            # Tüm client'ların zaman aralıklarını göster
            all_starts = [s.index[0] for s in trimmed_series.values()]
            all_ends = [s.index[-1] for s in trimmed_series.values()]
            console.print(f"   [dim]- Earliest start:[/dim] [green]{min(all_starts)}[/green]")
            console.print(f"   [dim]- Latest end:[/dim] [green]{max(all_ends)}[/green]")
        
        # Multi-Client LSTM Training
        console.print("\n" + "="*60)
        result, failed_clients = run_lstm_training(
            trimmed_series,
            console=console,
            tracker=tracker,
            error_handler=error_handler,
            sequence_length=24,
            forecast_horizon=1,
            embedding_dim=16,  # Increased for better client representation
            lstm_units=64,  # Increased for better learning capacity
            dropout_rate=0.2,
            learning_rate=0.002,  # Increased for faster learning
            num_layers=2,  # Increased for deeper learning
            batch_size=2048,  # Increased for much faster training
            epochs=30
        )
        console.print("="*60)
        
        if result:
            lstm_results = result
            # Update tracker with client results for the report
            if tracker:
                # Since we trained one global model, we mark all clients as successful
                # with the global metrics (or we could compute per-client if needed)
                for client_id in trimmed_series.keys():
                    tracker.client_results[client_id] = {
                        'status': 'success', 
                        'metrics': result.get('metrics', {})
                    }
        else:
            lstm_results = None
        
        # Finalize experiment and generate report
        console.print("\n[bold]Generating Experiment Report...[/bold]")
        summary = tracker.finalize()
        
        # Generate HTML report
        report_generator = ReportGenerator(tracker)
        report_path = report_generator.generate_report()
        
        console.print(f"\n[green][bold]Analysis completed successfully![/bold][/green]")
        console.print(f"[dim]The dataset has been transformed to remove Cold Start problems.[/dim]")
        console.print(f"\n[bold]Comprehensive Report Generated:[/bold]")
        console.print(f"   [cyan]{report_path}[/cyan]")
        console.print(f"\n[bold]Experiment Data:[/bold]")
        console.print(f"   [cyan]{tracker.experiment_dir}[/cyan]")
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"   • Total Clients: {summary.get('total_clients', 'N/A')}")
        if lstm_results:
            console.print(f"   • Model Metrics - MAE: {lstm_results.get('metrics', {}).get('mae', 'N/A'):.2f} kW")
            console.print(f"   • Model Metrics - RMSE: {lstm_results.get('metrics', {}).get('rmse', 'N/A'):.2f} kW")
            console.print(f"   • Model Metrics - MAPE (Safe): {lstm_results.get('metrics', {}).get('mape', 'N/A'):.2f}%")
            console.print(f"   • Model Metrics - WMAPE: {lstm_results.get('metrics', {}).get('wmape', 'N/A'):.2f}%")
        console.print(f"   • Errors: {summary.get('total_errors', 0)} (Recovered: {summary.get('recovered_errors', 0)})")

    except FileNotFoundError:
        error_msg = f"Data file not found: {data_path}"
        console.print(f"[red]   Error: {error_msg}[/red]")
        if tracker:
            tracker.log_error('FileNotFoundError', error_msg, {}, recovered=False)
            tracker.finalize()
        sys.exit(1)
    except ValueError as ve:
        error_msg = str(ve)
        console.print(f"[red]   Error: {error_msg}[/red]")
        if tracker:
            tracker.log_error('ValueError', error_msg, {}, recovered=False)
            tracker.finalize()
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        console.print(f"[red]   Error: {error_msg}[/red]")
        if tracker:
            tracker.log_error(type(e).__name__, error_msg, {}, recovered=False)
            # Try to generate report even if there was an error
            try:
                summary = tracker.finalize()
                report_generator = ReportGenerator(tracker)
                report_path = report_generator.generate_report()
                console.print(f"\n[yellow]Partial report generated: {report_path}[/yellow]")
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()
