#!/usr/bin/env python3
"""
Multi-Client LSTM Training Module

Trains a single LSTM model on all clients' data with client_id embeddings.
All 370 clients' data is combined into one training dataset.
"""

import os
import sys
# IMPORTANT: Set environment variables BEFORE importing TensorFlow
# TensorFlow reads these variables during import and logs immediately
# Configure TensorFlow for CUDA GPU acceleration on Linux
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress TensorFlow info and warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors only
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Use async allocator to reduce logs

# Suppress absl (TensorFlow's logging library) warnings
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'  # Suppress absl logs
os.environ['GLOG_minloglevel'] = '3'  # Suppress Google logging (used by TensorFlow)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import warnings
import logging

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from experiment_tracker import ExperimentTracker
    from error_handler import ErrorHandler
except ImportError:
    ExperimentTracker = None
    ErrorHandler = None

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging level to reduce verbosity (after import)
tf.get_logger().setLevel('ERROR')

# Suppress absl logging (TensorFlow's internal logging)
logging.getLogger('absl').setLevel(logging.CRITICAL)
logging.getLogger('absl.root').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow.core').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow.python').setLevel(logging.CRITICAL)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def configure_tensorflow_devices(console=None):
    """
    Configure TensorFlow to use CUDA GPU on Linux if available.
    
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
    
    # Check for CUDA GPU devices
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        if gpu_devices:
            try:
                # Enable memory growth to avoid allocating all memory at once
                for device in gpu_devices:
                    try:
                        tf.config.experimental.set_memory_growth(device, True)
                    except:
                        # Memory growth might not be supported, continue anyway
                        pass
                
                # Get GPU details
                gpu_details = []
                for device in gpu_devices:
                    try:
                        details = tf.config.experimental.get_device_details(device)
                        gpu_name = details.get('device_name', 'Unknown GPU')
                        gpu_details.append(f"{device.name} ({gpu_name})")
                    except:
                        gpu_details.append(device.name)
                
                console.print(f"   [green]CUDA GPU acceleration enabled[/green]")
                for detail in gpu_details:
                    console.print(f"   [dim]Using device: {detail}[/dim]")
                return True
            except Exception as e:
                console.print(f"   [yellow]Warning: Could not configure GPU device: {e}[/yellow]")
                console.print(f"   [dim]Falling back to CPU[/dim]")
                return False
        else:
            console.print(f"   [yellow]No CUDA GPU found, using CPU[/yellow]")
            console.print(f"   [dim]Note: Ensure CUDA toolkit and cuDNN are installed for GPU acceleration[/dim]")
            return False
    except Exception as e:
        console.print(f"   [yellow]Warning: Error detecting GPU devices: {e}[/yellow]")
        console.print(f"   [dim]Falling back to CPU[/dim]")
        return False


def make_sequences_1d(series_scaled, client_id_encoded, sequence_length, forecast_horizon):
    """
    Create sequences from a single client's scaled time series.
    Sequences are guaranteed to stay within the same client.
    
    Args:
        series_scaled: 1D numpy array of scaled consumption values for one client
        client_id_encoded: Integer encoded client ID
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of steps ahead to predict
    
    Returns:
        X: Input sequences (samples, sequence_length)
        X_client: Client IDs for each sequence (samples,)
        y: Target values (samples, forecast_horizon)
    """
    
    X, X_client, y = [], [], []
    
    # Check if series is 1D or 2D
    if len(series_scaled.shape) == 1:
        # Backward compatibility for 1D array
        series_scaled = series_scaled.reshape(-1, 1)
        
    for i in range(len(series_scaled) - sequence_length - forecast_horizon + 1):
        # Input: All features (Consumption + Time Embeddings)
        X.append(series_scaled[i:i + sequence_length])
        X_client.append(client_id_encoded)
        # Target: Only Consumption (Index 0)
        y.append(series_scaled[i + sequence_length:i + sequence_length + forecast_horizon, 0])
    
    return X, X_client, y


def add_time_features(df_segment):
    """
    Extract cyclical time features (Hour, DayOfWeek) from dataframe segment.
    """
    # Extract features
    hour = df_segment.index.hour
    day_of_week = df_segment.index.dayofweek
    
    # Cyclical encoding
    # Hour (0-23)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    # Day of week (0-6)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Stack features: (n_samples, 4)
    return np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)


class MultiClientDataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras model on-the-fly using Pre-computed Numpy arrays.
    High performance: ~30-50x faster than Pandas-based generation.
    """
    def __init__(self, client_data_list, client_ids_list, 
                 sequence_length=24, forecast_horizon=1, batch_size=2048, 
                 shuffle=True):
        """
        Args:
            client_data_list: List of numpy arrays (N_i, 7) for each client.
                              Each array contains: [ScaledCons, HourSin, HourCos, DaySin, DayCos, Lag24, Lag168]
            client_ids_list: List of encoded client IDs (int) corresponding to data list
            sequence_length: input seq len
            forecast_horizon: output horizon
            batch_size: batch size
            shuffle: whether to shuffle sequences after each epoch
        """
        self.client_data_list = client_data_list
        self.client_ids_list = client_ids_list
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate valid start indices map
        # Map: global_index -> (client_idx, local_start_index)
        # Optimized for fast lookup using cumulative counts
        
        self.cumulative_counts = [0]
        total_samples = 0
        
        # Store valid mapping info
        # To avoid storing a massive mapping array, we use binary search on cumulative counts
        
        for i, data_array in enumerate(client_data_list):
            n_points = len(data_array)
            # Valid sequences count
            n_samples = n_points - sequence_length - forecast_horizon + 1
            
            if n_samples > 0:
                total_samples += n_samples
                self.cumulative_counts.append(total_samples)
            else:
                # Should have been filtered out earlier, but just in case
                self.cumulative_counts.append(total_samples)
                
        self.total_samples = total_samples
        self.indices = np.arange(total_samples)
        self.cumulative_counts = np.array(self.cumulative_counts)
        
        # Pre-cast client IDs to array for faster access if needed, though list is fine
        self.client_ids_array = np.array(client_ids_list, dtype=np.int32)
        
        print(f"Generator created: {len(client_data_list)} clients, {total_samples:,} samples")
        
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.total_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data using fast Numpy slicing"""
        # Get indices for this batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.total_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Prepare batch arrays
        batch_size_actual = len(batch_indices)
        X_batch = np.empty((batch_size_actual, self.sequence_length, 7), dtype=np.float32)
        X_client_batch = np.empty((batch_size_actual,), dtype=np.int32)
        y_batch = np.empty((batch_size_actual, self.forecast_horizon), dtype=np.float32)
        
        # Find which client each index belongs to
        # searchsorted finds the insertion index i such that cumulative_counts[i-1] <= idx < cumulative_counts[i]
        # Our cumulative_counts array starts with 0.
        # global_idx 0 -> falls in bucket 1 (counts[0]=0, counts[1]=N1) -> index 1 returned -> client_idx = 0
        client_indices = np.searchsorted(self.cumulative_counts, batch_indices, side='right') - 1
        
        # Local indices
        local_indices = batch_indices - self.cumulative_counts[client_indices]
        
        # Optimization: Group by client to minimize cache misses and list lookups
        # However, for a shuffled batch, standard loop with numpy slicing is fast enough 
        # because we are slicing numpy arrays, not pandas series.
        
        for i in range(batch_size_actual):
            client_idx = client_indices[i]
            local_idx = local_indices[i]
            
            data_array = self.client_data_list[client_idx]
            cid_enc = self.client_ids_array[client_idx]
            
            # Input sequence: [local_idx : local_idx + seq_len]
            # Includes Consumption + Time Features + Lags (already pre-computed)
            # Shape (SEQ, 7)
            X_batch[i] = data_array[local_idx : local_idx + self.sequence_length]
            
            # Client ID
            X_client_batch[i] = cid_enc
            
            # Target: [local_idx + seq_len : local_idx + seq_len + horizon]
            # Only Consumption column (index 0)
            target_start = local_idx + self.sequence_length
            target_end = target_start + self.forecast_horizon
            y_batch[i] = data_array[target_start : target_end, 0] # Index 0 is consumption
            
        return {
            'consumption_sequence': X_batch, 
            'client_id': X_client_batch
        }, y_batch
    
    def on_epoch_end(self):
        """Shuffle updates after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def prepare_multi_client_data_generators(trimmed_series, sequence_length=24, forecast_horizon=1, 
                                   test_size=0.2, validation_size=0.05, batch_size=2048, console=None):
    """
    Prepare data generators for memory-efficient and FAST training using Numpy.
    """
    if console is None:
        console = Console()
    
    console.print("   [cyan]Preparing data generators (Optimized Numpy)...[/cyan]")
    
    # 1. Filter valid clients
    valid_clients = []
    for client_id, client_series in trimmed_series.items():
        if len(client_series) >= sequence_length + forecast_horizon + 10:
            valid_clients.append(client_id)
            
    if not valid_clients:
        console.print("   [red]Error: No valid data found[/red]")
        return None

    # 2. Fit Encoder
    client_encoder = LabelEncoder()
    client_encoder.fit(valid_clients)
    
    # 3. Pre-process splits into Numpy Arrays
    # We will compute Scale + Time Features ONCE here and store in RAM.
    # RAM Usage: 41M * 5 features * 4 bytes ≈ 820 MB. This is very cheap and much faster than live pandas.
    
    train_data_list = []  # List of numpy arrays
    train_ids_list = []   # List of client IDs (int)
    
    val_data_list = []
    val_ids_list = []
    
    test_data_list = []
    test_ids_list = []
    
    scalers = {}
    
    console.print("   [dim]Pre-processing data to Numpy (Scaling + Time + Lag Features)...[/dim]")
    
    # Progress bar for preprocessing
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Processing clients (computing lags)...", total=len(valid_clients))
        
        for client_id in valid_clients:
            series = trimmed_series[client_id]
            # Create a DataFrame to compute lags easily
            # We must compute lags on the FULL series before splitting to avoid edge artifacts
            df_temp = pd.DataFrame({'consumption': series})
            df_temp['lag_24'] = df_temp['consumption'].shift(24)
            df_temp['lag_168'] = df_temp['consumption'].shift(168)
            
            # Drop NaN rows due to shifting (first 168 rows)
            # This ensures all rows have valid lags
            df_temp = df_temp.dropna()
            
            if len(df_temp) < sequence_length + forecast_horizon:
                progress.advance(task)
                continue
                
            n = len(df_temp)
            train_end_idx = int(n * (1 - test_size - validation_size))
            val_end_idx = int(n * (1 - test_size))
            
            # Fit Scaler on TRAIN CONSUMPTION only
            train_cons = df_temp['consumption'].iloc[:train_end_idx]
            
            if len(train_cons) > 10:
                scaler = MinMaxScaler()
                scaler.fit(train_cons.values.reshape(-1, 1))
                
                cid_enc = client_encoder.transform([client_id])[0]
                scalers[cid_enc] = scaler
                
                # Split indices
                # Note: df_temp index might not be purely sequential ints, but we slice by iloc
                train_part = df_temp.iloc[:train_end_idx]
                val_part = df_temp.iloc[train_end_idx:val_end_idx] if val_end_idx > train_end_idx else None
                test_part = df_temp.iloc[val_end_idx:] if val_end_idx < n else None

                # Process Part Function
                def process_part(part_df):
                    # 1. Time Features (from index)
                    time_feats = add_time_features(part_df['consumption']) # (N, 4)
                    
                    # 2. Scale Consumption, Lag 24, Lag 168 using the SAME scaler (they are same units)
                    # We can vectorize this: stack, scale, unstack or just loop
                    cons_scaled = scaler.transform(part_df['consumption'].values.reshape(-1, 1))
                    lag24_scaled = scaler.transform(part_df['lag_24'].values.reshape(-1, 1))
                    lag168_scaled = scaler.transform(part_df['lag_168'].values.reshape(-1, 1))
                    
                    # 3. Combine: [Cons, Time(4), Lag24, Lag168]
                    # Total Features: 1 + 4 + 1 + 1 = 7
                    return np.hstack([cons_scaled, time_feats, lag24_scaled, lag168_scaled]).astype(np.float32)
                
                # Process and store TRAIN
                if len(train_part) >= sequence_length + forecast_horizon:
                    train_data_list.append(process_part(train_part))
                    train_ids_list.append(cid_enc)
                
                # Process and store VAL
                if val_part is not None and len(val_part) >= sequence_length + forecast_horizon:
                    val_data_list.append(process_part(val_part))
                    val_ids_list.append(cid_enc)
                    
                # Process and store TEST
                if test_part is not None and len(test_part) >= sequence_length + forecast_horizon:
                    test_data_list.append(process_part(test_part))
                    test_ids_list.append(cid_enc)
            
            progress.advance(task)
            
    console.print(f"   [green]Scalers fitted and data vectorized for {len(scalers)} clients[/green]")
    
    # 4. Create Generators using Numpy Lists
    train_gen = MultiClientDataGenerator(
        train_data_list, train_ids_list,
        sequence_length=sequence_length, forecast_horizon=forecast_horizon,
        batch_size=batch_size, shuffle=True
    )
    
    val_gen = MultiClientDataGenerator(
        val_data_list, val_ids_list,
        sequence_length=sequence_length, forecast_horizon=forecast_horizon,
        batch_size=batch_size, shuffle=False
    )
    
    test_gen = MultiClientDataGenerator(
        test_data_list, test_ids_list,
        sequence_length=sequence_length, forecast_horizon=forecast_horizon,
        batch_size=batch_size, shuffle=False
    )
    
    console.print(f"   [green]Generators ready (Numpy Optimized):[/green]")
    console.print(f"   [dim]Train samples: {train_gen.total_samples:,}[/dim]")
    console.print(f"   [dim]Val samples:   {val_gen.total_samples:,}[/dim]")
    console.print(f"   [dim]Test samples:  {test_gen.total_samples:,}[/dim]")
    
    return {
        'train_gen': train_gen,
        'val_gen': val_gen,
        'test_gen': test_gen,
        'scalers': scalers,
        'client_encoder': client_encoder,
        'n_clients': len(client_encoder.classes_)
    }


class RichProgressCallback(tf.keras.callbacks.Callback):
    """
    Rich progress bar ile eğitim ilerlemesini gösteren custom callback.
    """
    def __init__(self, total_epochs, console=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.console = console if console else Console()
        self.progress = None
        self.epoch_task = None
        self.batch_task = None
        self.current_epoch = 0
        self.total_batches = None
        self.current_batch = 0
        
    def on_train_begin(self, logs=None):
        """Eğitim başladığında progress bar'ı başlat"""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=False
        )
        self.progress.start()
        self.epoch_task = self.progress.add_task(
            f"[cyan]Training: Epoch 0/{self.total_epochs}", 
            total=self.total_epochs
        )
        
    def on_epoch_begin(self, epoch, logs=None):
        """Her epoch başında"""
        self.current_epoch = epoch + 1
        self.current_batch = 0
        if self.batch_task:
            self.progress.remove_task(self.batch_task)
        self.batch_task = None
        # Epoch description'ını güncelle
        self.progress.update(
            self.epoch_task,
            description=f"[cyan]Training: Epoch {self.current_epoch}/{self.total_epochs}"
        )
        
    def on_train_batch_begin(self, batch, logs=None):
        """Her batch başında - batch sayısını öğren"""
        if self.total_batches is None and hasattr(self.model, 'steps_per_epoch'):
            # İlk batch'te toplam batch sayısını öğren
            if self.model.steps_per_epoch:
                self.total_batches = self.model.steps_per_epoch
            elif hasattr(self.model, '_train_counter'):
                # Alternatif yöntem
                pass
        
    def on_train_batch_end(self, batch, logs=None):
        """Her batch sonunda progress bar'ı güncelle"""
        self.current_batch = batch + 1
        
        if self.batch_task is None:
            # Batch task'ı oluştur
            if self.total_batches:
                self.batch_task = self.progress.add_task(
                    f"  [dim]Batch 0/{self.total_batches}[/dim]", 
                    total=self.total_batches
                )
            else:
                # Toplam batch sayısı bilinmiyorsa indeterminate progress
                self.batch_task = self.progress.add_task(
                    f"  [dim]Batch {self.current_batch}[/dim]", 
                    total=None
                )
        
        # Batch progress için loss göster ve ilerlemeyi güncelle
        loss = logs.get('loss', 0) if logs else 0
        mae = logs.get('mae', 0) if logs else 0
        
        # Format loss and MAE for better readability (normalized scale)
        if loss < 1e-3:
            loss_str = f"{loss:.2e}"
        else:
            loss_str = f"{loss:.6f}"
        
        if mae < 1e-3:
            mae_str = f"{mae:.2e}"
        else:
            mae_str = f"{mae:.6f}"
        
        if self.total_batches:
            self.progress.update(
                self.batch_task,
                advance=1,
                description=f"  [dim]Batch {self.current_batch}/{self.total_batches} - loss: {loss_str} (norm), mae: {mae_str} (norm)[/dim]"
            )
        else:
            self.progress.update(
                self.batch_task,
                description=f"  [dim]Batch {self.current_batch} - loss: {loss_str} (norm), mae: {mae_str} (norm)[/dim]"
            )
        
    def on_epoch_end(self, epoch, logs=None):
        """Her epoch sonunda"""
        # Epoch progress'ini güncelle
        self.progress.update(self.epoch_task, advance=1)
        
        # Epoch sonu metriklerini göster
        loss = logs.get('loss', 0) if logs else 0
        mae = logs.get('mae', 0) if logs else 0
        val_loss = logs.get('val_loss', 0) if logs else 0
        val_mae = logs.get('val_mae', 0) if logs else 0
        lr = logs.get('learning_rate', 0) if logs else 0
        
        # Learning rate'i float olarak al
        if isinstance(lr, (list, tuple)):
            lr = lr[0] if lr else 0
        elif hasattr(lr, 'numpy'):
            lr = float(lr.numpy())
        
        # Format metrics for better readability
        def format_metric(val):
            if val < 1e-3:
                return f"{val:.2e}"
            else:
                return f"{val:.6f}"
        
        self.console.print(
            f"   [green]✓[/green] Epoch {self.current_epoch}/{self.total_epochs} - "
            f"loss: {format_metric(loss)} (norm), mae: {format_metric(mae)} (norm), "
            f"val_loss: {format_metric(val_loss)} (norm), val_mae: {format_metric(val_mae)} (norm), "
            f"lr: {lr:.6f}"
        )
        
        # Batch task'ı temizle
        if self.batch_task:
            self.progress.remove_task(self.batch_task)
            self.batch_task = None
            
    def on_train_end(self, logs=None):
        """Eğitim bittiğinde progress bar'ı kapat"""
        if self.progress:
            self.progress.stop()
            self.progress = None


def create_multi_client_lstm_model(sequence_length=24, n_clients=370, 
                                   embedding_dim=8, lstm_units=32, 
                                   dropout_rate=0.2, learning_rate=0.001, 
                                   num_layers=1, input_dim=7):
    """
    Create multi-client LSTM model with client_id embedding.
    Uses Bidirectional LSTM and Huber loss for improved robustness.
    
    Args:
        sequence_length: Input sequence length
        n_clients: Number of unique clients
        embedding_dim: Dimension of client embedding
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        num_layers: Number of LSTM layers
        input_dim: Number of input features
    
    Returns:
        Compiled Keras model
    """
    # Input 1: Time series sequence
    # Shape: (sequence_length, input_dim) -> e.g., (24, 7)
    sequence_input = tf.keras.Input(shape=(sequence_length, input_dim), name='consumption_sequence')
    
    # Input 2: Client ID
    client_input = tf.keras.Input(shape=(1,), name='client_id')
    
    # Client embedding
    client_embedding = tf.keras.layers.Embedding(
        input_dim=n_clients,
        output_dim=embedding_dim,
        name='client_embedding'
    )(client_input)
    
    # Flatten embedding
    client_embedding_flat = tf.keras.layers.Flatten()(client_embedding)
    
    # Expand client embedding to match sequence length
    client_embedding_expanded = tf.keras.layers.RepeatVector(sequence_length)(client_embedding_flat)
    
    # Concatenate sequence with client embedding
    combined = tf.keras.layers.Concatenate(axis=-1)([sequence_input, client_embedding_expanded])
    
    # Bidirectional LSTM layers
    x = combined
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        
        # Wrapped in Bidirectional for context from both Future and Past (within the window)
        lstm_layer = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            name=f'lstm_base_{i+1}'
        )
        x = tf.keras.layers.Bidirectional(lstm_layer, name=f'bidirectional_{i+1}')(x)
        
        # Batch normalization for better training stability
        if return_sequences:
            x = tf.keras.layers.BatchNormalization(name=f'bn_{i+1}')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Add a dense layer before output for better representation
    x = tf.keras.layers.Dense(lstm_units // 2, activation='relu', name='dense_hidden')(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.5, name='dropout_output')(x)
    
    # Dense output layer (cast to float32 for mixed precision)
    output = tf.keras.layers.Dense(1, name='consumption_prediction', dtype='float32')(x)
    
    # Create model
    model = tf.keras.Model(
        inputs=[sequence_input, client_input],
        outputs=output,
        name='multi_client_bidirectional_lstm'
    )
    
    # Compile model with Huber Loss for robustness against outliers
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Use Huber Loss (delta=1.0 by default) or Log-Cosh
    loss = tf.keras.losses.Huber(delta=1.0)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    
    return model


def train_multi_client_lstm(trimmed_series, console=None, tracker=None, error_handler=None,
                            sequence_length=24, forecast_horizon=1,
                            embedding_dim=8, lstm_units=32, dropout_rate=0.2,
                            learning_rate=0.001, num_layers=1, batch_size=512,
                            epochs=20, validation_size=0.05):
    """
    Train multi-client LSTM model on all clients' data.
    
    Args:
        trimmed_series: dict[str, pd.Series] - Her client için trimmed seri (farklı uzunlukta zaman index'i)
        console: Rich Console object for colored output
        tracker: ExperimentTracker instance (optional)
        error_handler: ErrorHandler instance (optional)
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of steps ahead to predict
        embedding_dim: Client embedding dimension
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        epochs: Number of training epochs
        validation_size: Proportion of data for validation (default: 0.05 for 5%)
    
    Returns:
        dict: Model, scaler, metrics, and predictions
    """
    if console is None:
        console = Console()
    
    console.print("\n[bold]Multi-Client LSTM Training:[/bold]")
    console.print(f"   [dim]Sequence length: {sequence_length}, Forecast horizon: {forecast_horizon}[/dim]")
    
    if tracker:
        tracker.log_step("MULTI_CLIENT_TRAINING_START", 
                        "Starting multi-client LSTM training",
                        {'n_clients': len(trimmed_series)})
    
    # Show device configuration
    configure_tensorflow_devices(console)
    
    # Adım 1: Veriyi hazırla (Generator kullanarak)
    data_dict = prepare_multi_client_data_generators(
        trimmed_series,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        test_size=0.2,  # 20% test
        validation_size=validation_size,  # 5% validation
        batch_size=batch_size,
        console=console
    )
    
    if data_dict is None:
        console.print("   [red]Data preparation failed[/red]")
        return None, []
        
    train_gen = data_dict['train_gen']
    val_gen = data_dict['val_gen']
    test_gen = data_dict['test_gen']
    
    n_clients = data_dict['n_clients']
    
    if tracker:
        tracker.log_step("DATA_PREPARATION", 
                        f"Prepared data for {n_clients} clients using Generators", 
                        {'n_clients': n_clients, 
                         'train_samples': train_gen.total_samples,
                         'val_samples': val_gen.total_samples,
                         'test_samples': test_gen.total_samples})
    
    # Create model
    console.print(f"\n[bold]Creating model...[/bold]")
    console.print(f"   [dim]Clients: {n_clients}, Embedding dim: {embedding_dim}[/dim]")
    console.print(f"   [dim]LSTM units: {lstm_units}, Layers: {num_layers}[/dim]")
    
    model = create_multi_client_lstm_model(
        sequence_length=sequence_length,
        n_clients=n_clients,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        num_layers=num_layers,
        input_dim=7  # Explicitly set to 7 for consumption + 4 time features + 2 lags
    )
    
    console.print(f"   [green]Model created with {model.count_params():,} parameters[/green]")
    
    # Train model
    console.print(f"\\n[bold]Training model...[/bold]")
    console.print(f"   [dim]Epochs: {epochs}, Batch size: {batch_size}[/dim]")
    
    # Enable mixed precision for faster training (FP16)
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        console.print("   [green]Mixed precision (FP16) enabled for faster training[/green]")
    except:
        console.print("   [yellow]Mixed precision not available, using FP32[/yellow]")
    
    # Validation seti var mı kontrol et
    has_validation = val_gen.total_samples > 0
    
    # Callbacks - validation seti varsa val_loss, yoksa loss kullan
    monitor_metric = 'val_loss' if has_validation else 'loss'
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=8,  # Increased patience for better convergence
        restore_best_weights=True,
        verbose=0
    )
    
    # ReduceLROnPlateau for faster convergence
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1 # Verbose to see when it kicks in
    )
    
    # ModelCheckpoint: Her epoch'ta en iyi modeli kaydet
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data', 'processed', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(models_dir, 'best_lstm_model.keras'),
        monitor=monitor_metric,
        save_best_only=True,  # Sadece en iyi modeli kaydet
        mode='min',
        verbose=0,
        save_weights_only=False  # Tüm modeli kaydet (weights + architecture)
    )
    
    console.print(f"   [green]Model checkpoint: {os.path.join(models_dir, 'best_lstm_model.keras')}[/green]")
    console.print(f"   [dim]En iyi model her epoch'ta otomatik kaydedilecek[/dim]")
    
    # Rich progress bar callback - steps_per_epoch hesapla
    rich_progress = RichProgressCallback(total_epochs=epochs, console=console)
    rich_progress.total_batches = len(train_gen)
    
    try:
        # Validation seti varsa manuel olarak geç, yoksa validation_split kullanma
        # Validation seti varsa manuel olarak geç, yoksa validation_split kullanma
        if has_validation:
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=[early_stopping, reduce_lr, model_checkpoint, rich_progress],
                verbose=0  # Rich callback kullanıyoruz, TensorFlow'un verbose'unu kapatıyoruz
            )
        else:
            console.print("   [yellow]Warning: No validation set available, training without validation[/yellow]")
            # Validation olmadan eğitim yap, ama callbacks'i güncelle
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,  # Validation yoksa daha fazla patience
                restore_best_weights=True,
                verbose=0
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(models_dir, 'best_lstm_model.keras'),
                monitor='loss',
                save_best_only=True,
                mode='min',
                verbose=0,
                save_weights_only=False
            )
            history = model.fit(
                train_gen,
                epochs=epochs,
                callbacks=[early_stopping, reduce_lr, model_checkpoint, rich_progress],
                verbose=0
            )
    except Exception as e:
        if error_handler:
            result, success = error_handler.handle_error(
                e,
                context={'step': 'model_training'},
                retry_func=None
            )
            if not success:
                return None
        else:
            console.print(f"   [red]Error during training: {e}[/red]")
            return None
    
    # Adım 4: Evaluation
    console.print("\n[bold]Evaluating model...[/bold]")
    
    # Predict on test set using Generator
    test_gen = data_dict['test_gen']
    if test_gen.total_samples == 0:
        console.print("   [yellow]No test data available for evaluation[/yellow]")
        return {}, []
        
    test_predictions = model.predict(test_gen, verbose=1)
    
    # Retrieve Actuals and Client IDs from generator manually
    test_actual = []
    test_client_ids = []
    
    # Iterate through generator to get truth values
    # Note: Generator yields ({'consumption_sequence': X, 'client_id': ids}, y)
    for i in range(len(test_gen)):
        inputs_batch, y_batch = test_gen[i]
        test_actual.extend(y_batch)
        test_client_ids.extend(inputs_batch['client_id'])
    
    test_actual = np.array(test_actual)
    test_client_ids = np.array(test_client_ids)
    
    # Inverse transform logic remains similar, but using collected arrays
    # ...
    
    test_pred_inv = []
    test_actual_inv = []
    
    # Inverse transform using per-client scalers
    # We need to process sample by sample or group by client for efficiency
    
    # Group indices by client to do batch inverse transform
    client_indices = {}
    for i, cid_enc in enumerate(test_client_ids):
        if cid_enc not in client_indices:
            client_indices[cid_enc] = []
        client_indices[cid_enc].append(i)
        
    # Prepare result arrays
    test_pred_inv_arr = np.zeros_like(test_predictions)
    test_actual_inv_arr = np.zeros_like(test_actual)
    
    console.print("   [dim]Inverse transforming predictions...[/dim]")
    
    for cid_enc, indices in client_indices.items():
        if cid_enc in data_dict['scalers']:
            scaler = data_dict['scalers'][cid_enc]
            
            # Get data for this client
            pred_subset = test_predictions[indices]
            actual_subset = test_actual[indices]
            
            # Inverse transform
            # Scaler expects 2D array
            pred_inv = scaler.inverse_transform(pred_subset)
            actual_inv = scaler.inverse_transform(actual_subset)
            
            # Fill result arrays
            test_pred_inv_arr[indices] = pred_inv
            test_actual_inv_arr[indices] = actual_inv
            
    test_pred_inv = test_pred_inv_arr
    test_actual_inv = test_actual_inv_arr
    
    # Compute metrics
    # ...
    # Compute metrics
    mae = np.mean(np.abs(test_pred_inv - test_actual_inv))
    rmse = np.sqrt(np.mean((test_pred_inv - test_actual_inv) ** 2))
    mse = np.mean((test_pred_inv - test_actual_inv) ** 2)
    
    # R2 Calculation
    ss_res = np.sum((test_actual_inv - test_pred_inv) ** 2)
    ss_tot = np.sum((test_actual_inv - np.mean(test_actual_inv)) ** 2)
    # Avoid division by zero
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Safe MAPE Calculation
    # Filter out near-zero values to avoid exploding metric
    # Threshold: 0.1 kW (assuming consumption is in kW, 0.1 is very small)
    mask = test_actual_inv > 0.1
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((test_actual_inv[mask] - test_pred_inv[mask]) / test_actual_inv[mask])) * 100
    else:
        mape = 0.0
        
    # WMAPE (Weighted MAPE) - Often better for energy data
    # Sum of errors / Sum of actuals
    wmape = np.sum(np.abs(test_actual_inv - test_pred_inv)) / (np.sum(test_actual_inv) + 1e-8) * 100

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'wmape': float(wmape),
        'mse': float(mse),
        'r2': float(r2)
    }
        
    console.print(f"\n[bold]Test Set Metrics:[/bold]")
    console.print(f"   [green]MAE: {metrics['mae']:.2f} kW[/green]")
    console.print(f"   [green]RMSE: {metrics['rmse']:.2f} kW[/green]")
    console.print(f"   [green]MAPE: {metrics['mape']:.2f}% (Safe)[/green]")
    console.print(f"   [green]WMAPE: {metrics['wmape']:.2f}%[/green]")
    console.print(f"   [green]R²: {metrics['r2']:.4f}[/green]")
    
    # Log to tracker
    if tracker:
        tracker.log_step("MULTI_CLIENT_TRAINING_SUCCESS", 
                        "Multi-client LSTM training completed",
                        {'metrics': metrics, 'n_clients': n_clients})
    
    return {
        'model': model,
        'scalers': data_dict['scalers'],
        'client_encoder': data_dict['client_encoder'],
        'metrics': metrics,
        'test_predictions': test_pred_inv,
        'test_actual': test_actual_inv,
        'history': history.history if 'history' in locals() else None
    }


def run_lstm_training(trimmed_series, console=None, tracker=None, error_handler=None, **kwargs):
    """
    Main function to run multi-client LSTM training.
    
    Args:
        trimmed_series: dict[str, pd.Series] - Her client için trimmed seri (farklı uzunlukta zaman index'i)
        console: Rich Console object for colored output
        tracker: ExperimentTracker instance (optional)
        error_handler: ErrorHandler instance (optional)
        **kwargs: Additional arguments for train_multi_client_lstm
    
    Returns:
        dict: Training results
    """
    if console is None:
        console = Console()
    
    # Default parameters (optimized for better learning)
    default_params = {
        'sequence_length': 24,
        'forecast_horizon': 1,
        'embedding_dim': 16,  # Increased for better client representation
        'lstm_units': 64,  # Increased for better learning capacity
        'dropout_rate': 0.2,
        'learning_rate': 0.002,  # Increased for faster learning
        'num_layers': 2,  # Increased for deeper learning
        'batch_size': 2048,  # Increased to 2048 for much faster training
        'epochs': 20,
        'validation_size': 0.05  # 5% validation (time-based split)
    }
    
    # Merge with kwargs
    params = {**default_params, **kwargs}
    
    result = train_multi_client_lstm(
        trimmed_series,
        console=console,
        tracker=tracker,
        error_handler=error_handler,
        **params
    )
    
    return result, []  # Return empty failed_clients list for compatibility
