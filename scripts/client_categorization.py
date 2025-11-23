#!/usr/bin/env python3
"""
Client Categorization Module

Categorizes clients into:
- Low Variance (Stable): Easy to predict, regular consumer
- High Variance (Irregular): Challenging consumer where Deep Learning shows potential
- Sparse Data: Consumer with intermittent data
"""


from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import matplotlib
# Set matplotlib backend to Agg (non-interactive) - works without tkinter
# We save plots to files, so interactive display is not required
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def calculate_variance_metrics(data):
    """
    Calculate variance and sparsity metrics for a client's data.
    
    Args:
        data: pandas Series with consumption values
    
    Returns:
        dict: Metrics including variance, coefficient of variation, sparsity ratio
    """
    # Remove NaN values
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return {
            'variance': 0,
            'std': 0,
            'mean': 0,
            'cv': 0,  # Coefficient of Variation
            'sparsity_ratio': 1.0,
            'data_points': 0,
            'total_points': len(data)
        }
    
    variance = clean_data.var()
    std = clean_data.std()
    mean = clean_data.mean()
    
    # Coefficient of Variation (CV) = std/mean (normalized variance)
    cv = std / mean if mean > 0 else 0
    
    # Sparsity: ratio of missing/zero values
    total_points = len(data)
    valid_points = len(clean_data)
    sparsity_ratio = 1.0 - (valid_points / total_points) if total_points > 0 else 1.0
    
    return {
        'variance': variance,
        'std': std,
        'mean': mean,
        'cv': cv,
        'sparsity_ratio': sparsity_ratio,
        'data_points': valid_points,
        'total_points': total_points
    }


def categorize_clients(df, console=None, 
                       low_variance_threshold=0.3,
                       high_variance_threshold=1.0,
                       sparse_threshold=0.3):
    """
    Categorize clients based on variance and sparsity metrics.
    
    Args:
        df: DataFrame with datetime index and client columns
        console: Rich Console object for colored output
        low_variance_threshold: CV threshold for low variance (default: 0.3)
        high_variance_threshold: CV threshold for high variance (default: 1.0)
        sparse_threshold: Sparsity ratio threshold (default: 0.3)
    
    Returns:
        dict: Categorized clients with statistics
    """
    if console is None:
        console = Console()
    
    console.print("\n[bold]Client Categorization Analysis:[/bold]")
    console.print("   [dim]Analyzing variance and sparsity patterns...[/dim]")
    
    client_metrics = {}
    
    # Calculate metrics for all clients
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Calculating metrics...", total=len(df.columns))
        
        for col in df.columns:
            metrics = calculate_variance_metrics(df[col])
            client_metrics[col] = metrics
            progress.update(task, advance=1)
    
    # Categorize clients
    low_variance_clients = []
    high_variance_clients = []
    sparse_clients = []
    other_clients = []
    
    for client_id, metrics in client_metrics.items():
        cv = metrics['cv']
        sparsity = metrics['sparsity_ratio']
        
        # Check sparsity first (highest priority)
        if sparsity >= sparse_threshold:
            sparse_clients.append({
                'client_id': client_id,
                'metrics': metrics
            })
        # Then check variance
        elif cv <= low_variance_threshold:
            low_variance_clients.append({
                'client_id': client_id,
                'metrics': metrics
            })
        elif cv >= high_variance_threshold:
            high_variance_clients.append({
                'client_id': client_id,
                'metrics': metrics
            })
        else:
            other_clients.append({
                'client_id': client_id,
                'metrics': metrics
            })
    
    # Calculate statistics
    total_clients = len(df.columns)
    
    stats = {
        'total_clients': total_clients,
        'low_variance': {
            'clients': low_variance_clients,
            'count': len(low_variance_clients),
            'percentage': (len(low_variance_clients) / total_clients * 100) if total_clients > 0 else 0
        },
        'high_variance': {
            'clients': high_variance_clients,
            'count': len(high_variance_clients),
            'percentage': (len(high_variance_clients) / total_clients * 100) if total_clients > 0 else 0
        },
        'sparse': {
            'clients': sparse_clients,
            'count': len(sparse_clients),
            'percentage': (len(sparse_clients) / total_clients * 100) if total_clients > 0 else 0
        },
        'other': {
            'clients': other_clients,
            'count': len(other_clients),
            'percentage': (len(other_clients) / total_clients * 100) if total_clients > 0 else 0
        }
    }
    
    return stats, client_metrics


def display_categorization_results(stats, client_metrics, console=None, title_suffix=""):
    """
    Display categorization results in a formatted table.
    
    Args:
        stats: Statistics dictionary from categorize_clients
        client_metrics: Dictionary with metrics for all clients
        console: Rich Console object for colored output
        title_suffix: Optional suffix for the table title
    """
    if console is None:
        console = Console()
    
    # Create summary table
    title = "Client Categorization Summary"
    if title_suffix:
        title = f"{title} {title_suffix}"
    table = Table(title=title, show_header=True, header_style="bold blue", expand=True)
    table.add_column("Category", style="cyan", width=25, no_wrap=False)
    table.add_column("Description", style="dim", width=45, no_wrap=False)
    table.add_column("Count", justify="right", style="green", width=8)
    table.add_column("Percentage", justify="right", style="yellow", width=12)
    
    table.add_row(
        "Low Variance (Stable)",
        "Easy to predict, regular consumer",
        str(stats['low_variance']['count']),
        f"{stats['low_variance']['percentage']:.2f}%"
    )
    
    table.add_row(
        "High Variance (Irregular)",
        "Challenging consumer, Deep Learning potential",
        str(stats['high_variance']['count']),
        f"{stats['high_variance']['percentage']:.2f}%"
    )
    
    table.add_row(
        "Sparse Data",
        "Consumer with intermittent data",
        str(stats['sparse']['count']),
        f"{stats['sparse']['percentage']:.2f}%"
    )
    
    table.add_row(
        "Other (Medium Variance)",
        "Moderate variance, regular patterns",
        str(stats['other']['count']),
        f"{stats['other']['percentage']:.2f}%"
    )
    
    table.add_row(
        "[bold]Total[/bold]",
        "[bold]All clients[/bold]",
        f"[bold]{stats['total_clients']}[/bold]",
        "[bold]100.00%[/bold]"
    )
    
    console.print("\n")
    console.print(table)
    
    # Create visualization
    try:
        # Create output directory for plots if it doesn't exist
        os.makedirs("data/processed/plots", exist_ok=True)
        
        # Prepare data for plotting
        categories = ['Low Variance', 'High Variance', 'Sparse Data', 'Other']
        counts = [
            stats['low_variance']['count'],
            stats['high_variance']['count'],
            stats['sparse']['count'],
            stats['other']['count']
        ]
        percentages = [
            stats['low_variance']['percentage'],
            stats['high_variance']['percentage'],
            stats['sparse']['percentage'],
            stats['other']['percentage']
        ]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Number of Clients', fontsize=12, fontweight='bold')
        ax1.set_title('Client Categorization - Count', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticklabels(categories, rotation=15, ha='right')
        
        # Add value labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(
            percentages, 
            labels=categories, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        ax2.set_title('Client Categorization - Percentage', fontsize=14, fontweight='bold')
        
        # Improve pie chart text visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"categorization{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_path = f"data/processed/plots/{plot_filename}"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        console.print(f"   [green]Visualization saved to: {plot_path}[/green]")
        plt.close()  # Close figure to free memory
    except Exception as e:
        console.print(f"   [yellow]Warning: Could not create visualization: {e}[/yellow]")


def analyze_client_categories(df, console=None, title_suffix=""):
    """
    Main function to analyze and categorize clients.
    
    Args:
        df: DataFrame with datetime index and client columns
        console: Rich Console object for colored output
        title_suffix: Optional suffix for the title (e.g., "(Before Trimming)" or "(After Trimming)")
    
    Returns:
        tuple: (stats, client_metrics)
    """
    if console is None:
        console = Console()
    
    # Categorize clients
    stats, client_metrics = categorize_clients(df, console)
    
    # Display results
    display_categorization_results(stats, client_metrics, console, title_suffix=title_suffix)
    
    return stats, client_metrics

