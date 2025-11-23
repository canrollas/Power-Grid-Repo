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
    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("Category", style="cyan", width=30)
    table.add_column("Description", style="dim", width=50)
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    
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

