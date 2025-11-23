#!/usr/bin/env python3
"""
LSTM Training Module with Optuna Hyperparameter Optimization

Trains LSTM models for electricity consumption prediction using trimmed data.
Uses Optuna for hyperparameter tuning.
"""


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import optuna
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import warnings
import os
warnings.filterwarnings('ignore')

# Configure TensorFlow for macOS Metal (MPS) backend
# Enable Metal Performance Shaders for GPU acceleration on Apple Silicon
os.environ['TF_METAL_DEVICE_PLACEMENT'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def configure_tensorflow_devices(console=None):
    """
    Configure TensorFlow to use Metal (MPS) on macOS if available.
    
    Args:
        console: Rich Console object for colored output
    """
    if console is None:
        console = Console()
    
    # List available physical devices
    physical_devices = tf.config.list_physical_devices()
    device_names = [d.name for d in physical_devices]
    device_types = [d.device_type for d in physical_devices]
    console.print(f"   [dim]Available devices: {device_names}[/dim]")
    console.print(f"   [dim]Device types: {device_types}[/dim]")
    
    # Check for Metal GPU (MPS) - on macOS with tensorflow-metal, GPU devices are MPS
    mps_devices = [d for d in physical_devices if d.device_type == 'GPU']
    
    # Also check if we can detect MPS backend
    try:
        # Try to get GPU device info
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            mps_devices = gpu_devices
    except:
        pass
    
    if mps_devices:
        try:
            # Enable memory growth to avoid allocating all memory at once
            for device in mps_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    # Memory growth might not be supported, continue anyway
                    pass
            console.print(f"   [green]Metal (MPS) GPU acceleration enabled[/green]")
            console.print(f"   [dim]Using device: {mps_devices[0].name}[/dim]")
            return True
        except Exception as e:
            console.print(f"   [yellow]Warning: Could not configure MPS device: {e}[/yellow]")
            console.print(f"   [dim]Falling back to CPU[/dim]")
            return False
    else:
        console.print(f"   [yellow]No Metal GPU found, using CPU[/yellow]")
        console.print(f"   [dim]Note: Install 'tensorflow-metal' for Apple Silicon GPU acceleration:[/dim]")
        console.print(f"   [dim]   pip install tensorflow-metal[/dim]")
        return False


def create_sequences(data, sequence_length, forecast_horizon=1):
    """
    Create sequences for time series prediction.
    
    Args:
        data: 1D array of time series data
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of steps ahead to predict (default: 1)
    
    Returns:
        X: Input sequences (samples, sequence_length, features)
        y: Target values (samples, forecast_horizon)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])
    return np.array(X), np.array(y)


def prepare_client_data(client_data, sequence_length=24, forecast_horizon=1, test_size=0.2, val_size=0.1):
    """
    Prepare client data for LSTM training.
    
    Args:
        client_data: pandas Series with consumption values
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of steps ahead to predict
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
    
    Returns:
        dict: Contains scaled data, scaler, and train/val/test splits
    """
    # Remove NaN values
    clean_data = client_data.dropna().values.reshape(-1, 1)
    
    if len(clean_data) < sequence_length + forecast_horizon + 10:
        return None
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(clean_data).flatten()
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)
    
    if len(X) < 10:
        return None
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), shuffle=False
    )
    
    val_split = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_split), shuffle=False
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'original_data': clean_data
    }


def create_lstm_model(sequence_length, n_features=1, 
                      lstm_units=50, dropout_rate=0.2, 
                      learning_rate=0.001, num_layers=1):
    """
    Create LSTM model architecture.
    
    Args:
        sequence_length: Input sequence length
        n_features: Number of features (default: 1 for univariate)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        num_layers: Number of LSTM layers
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential()
    
    # First LSTM layer
    model.add(tf.keras.layers.LSTM(
        units=lstm_units,
        return_sequences=(num_layers > 1),
        input_shape=(sequence_length, n_features)
    ))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Additional LSTM layers
    for _ in range(num_layers - 1):
        model.add(tf.keras.layers.LSTM(units=lstm_units, return_sequences=(_ < num_layers - 2)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Dense output layer
    model.add(tf.keras.layers.Dense(1))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def objective(trial, data_dict, sequence_length):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        data_dict: Dictionary with train/val/test data
        sequence_length: Input sequence length
    
    Returns:
        Validation loss (MAE)
    """
    # Suggest hyperparameters
    lstm_units = trial.suggest_int('lstm_units', 32, 128, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50, step=10)
    
    # Create model
    model = create_lstm_model(
        sequence_length=sequence_length,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        num_layers=num_layers
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        data_dict['X_train'],
        data_dict['y_train'],
        validation_data=(data_dict['X_val'], data_dict['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Return validation loss
    val_loss = min(history.history['val_loss'])
    
    # Clean up
    del model
    tf.keras.backend.clear_session()
    
    return val_loss


def train_lstm_for_client(client_id, client_data, sequence_length=24, 
                         forecast_horizon=1, n_trials=20, console=None):
    """
    Train LSTM model for a single client with Optuna optimization.
    
    Args:
        client_id: Client identifier
        client_data: pandas Series with consumption values
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of steps ahead to predict
        n_trials: Number of Optuna trials
        console: Rich Console object for colored output
    
    Returns:
        dict: Model, scaler, best parameters, and evaluation metrics
    """
    if console is None:
        console = Console()
    
    # Prepare data
    data_dict = prepare_client_data(
        client_data, 
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )
    
    if data_dict is None:
        return None
    
    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, data_dict, sequence_length),
        n_trials=n_trials,
        show_progress_bar=False
    )
    
    # Get best parameters
    best_params = study.best_params
    
    # Train final model with best parameters
    final_model = create_lstm_model(
        sequence_length=sequence_length,
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        num_layers=best_params['num_layers']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    final_model.fit(
        data_dict['X_train'],
        data_dict['y_train'],
        validation_data=(data_dict['X_val'], data_dict['y_val']),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate on test set
    test_predictions = final_model.predict(data_dict['X_test'], verbose=0)
    test_actual = data_dict['y_test']
    
    # Inverse transform
    test_pred_inv = data_dict['scaler'].inverse_transform(test_predictions)
    test_actual_inv = data_dict['scaler'].inverse_transform(test_actual)
    
    # Calculate metrics
    mae = np.mean(np.abs(test_pred_inv - test_actual_inv))
    rmse = np.sqrt(np.mean((test_pred_inv - test_actual_inv) ** 2))
    mape = np.mean(np.abs((test_actual_inv - test_pred_inv) / (test_actual_inv + 1e-8))) * 100
    
    return {
        'client_id': client_id,
        'model': final_model,
        'scaler': data_dict['scaler'],
        'best_params': best_params,
        'best_trial_value': study.best_value,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        },
        'test_predictions': test_pred_inv,
        'test_actual': test_actual_inv
    }


def train_lstm_models(trimmed_df, console=None, 
                     sequence_length=24, forecast_horizon=1,
                     n_trials=20, max_clients=None,
                     client_category_filter=None):
    """
    Train LSTM models for multiple clients.
    
    Args:
        trimmed_df: DataFrame with trimmed consumption data
        console: Rich Console object for colored output
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of steps ahead to predict
        n_trials: Number of Optuna trials per client
        max_clients: Maximum number of clients to process (None for all)
        client_category_filter: Filter clients by category (None for all)
    
    Returns:
        dict: Results for all clients
    """
    if console is None:
        console = Console()
    
    console.print("\n[bold]LSTM Training with Optuna Hyperparameter Optimization:[/bold]")
    console.print(f"   [dim]Sequence length: {sequence_length}, Forecast horizon: {forecast_horizon}[/dim]")
    console.print(f"   [dim]Optuna trials per client: {n_trials}[/dim]")
    
    # Show device configuration
    configure_tensorflow_devices(console)
    
    # Select clients to process
    clients_to_process = list(trimmed_df.columns)
    if max_clients:
        clients_to_process = clients_to_process[:max_clients]
    
    results = {}
    failed_clients = []
    
    # Process clients with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Training LSTM models...", total=len(clients_to_process))
        
        for client_id in clients_to_process:
            try:
                result = train_lstm_for_client(
                    client_id,
                    trimmed_df[client_id],
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                    n_trials=n_trials,
                    console=console
                )
                
                if result:
                    results[client_id] = result
                else:
                    failed_clients.append(client_id)
                    
            except Exception as e:
                failed_clients.append(client_id)
                console.print(f"   [yellow]Warning: Failed to train model for {client_id}: {str(e)}[/yellow]")
            
            progress.update(task, advance=1)
    
    # Summary
    console.print(f"\n[green]Successfully trained {len(results)} models[/green]")
    if failed_clients:
        console.print(f"[yellow]Failed to train {len(failed_clients)} models[/yellow]")
    
    return results, failed_clients


def display_training_results(results, console=None):
    """
    Display training results summary.
    
    Args:
        results: Dictionary with training results for all clients
        console: Rich Console object for colored output
    """
    if console is None:
        console = Console()
    
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return
    
    # Calculate aggregate metrics
    mae_values = [r['metrics']['mae'] for r in results.values()]
    rmse_values = [r['metrics']['rmse'] for r in results.values()]
    mape_values = [r['metrics']['mape'] for r in results.values()]
    
    # Create summary table
    table = Table(title="LSTM Training Results Summary", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Median", justify="right", style="green")
    table.add_column("Min", justify="right", style="yellow")
    table.add_column("Max", justify="right", style="red")
    
    table.add_row(
        "MAE (kW)",
        f"{np.mean(mae_values):.2f}",
        f"{np.median(mae_values):.2f}",
        f"{np.min(mae_values):.2f}",
        f"{np.max(mae_values):.2f}"
    )
    
    table.add_row(
        "RMSE (kW)",
        f"{np.mean(rmse_values):.2f}",
        f"{np.median(rmse_values):.2f}",
        f"{np.min(rmse_values):.2f}",
        f"{np.max(rmse_values):.2f}"
    )
    
    table.add_row(
        "MAPE (%)",
        f"{np.mean(mape_values):.2f}",
        f"{np.median(mape_values):.2f}",
        f"{np.min(mape_values):.2f}",
        f"{np.max(mape_values):.2f}"
    )
    
    console.print("\n")
    console.print(table)
    
    # Best and worst performing clients
    best_client = min(results.items(), key=lambda x: x[1]['metrics']['mae'])
    worst_client = max(results.items(), key=lambda x: x[1]['metrics']['mae'])
    
    console.print(f"\n[bold]Best performing client:[/bold] [green]{best_client[0]}[/green] (MAE: {best_client[1]['metrics']['mae']:.2f} kW)")
    console.print(f"[bold]Worst performing client:[/bold] [red]{worst_client[0]}[/red] (MAE: {worst_client[1]['metrics']['mae']:.2f} kW)")


def run_lstm_training(trimmed_df, console=None, **kwargs):
    """
    Main function to run LSTM training pipeline.
    
    Args:
        trimmed_df: DataFrame with trimmed consumption data
        console: Rich Console object for colored output
        **kwargs: Additional arguments for train_lstm_models
    
    Returns:
        dict: Training results
    """
    if console is None:
        console = Console()
    
    # Train models
    results, failed_clients = train_lstm_models(trimmed_df, console=console, **kwargs)
    
    # Display results
    if results:
        display_training_results(results, console)
    
    return results, failed_clients

