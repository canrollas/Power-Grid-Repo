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
    Generates data for Keras model on-the-fly to avoid OOM.
    """
    def __init__(self, client_data_dict, scalers, client_encoder, 
                 sequence_length=24, forecast_horizon=1, batch_size=256, 
                 shuffle=True, is_training=True):
        """
        Args:
            client_data_dict: {client_id: pd.Series} - Raw data per client
            scalers: {client_id_encoded: scaler} - Fitted scalers
            client_encoder: LabelEncoder
            sequence_length: input seq len
            forecast_horizon: output horizon
            batch_size: batch size
            shuffle: whether to shuffle sequences after each epoch
            is_training: if True, returns (X, y), else (X, y) but y might be used for eval
        """
        self.client_data_dict = client_data_dict
        self.scalers = scalers
        self.client_encoder = client_encoder
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_training = is_training
        
        # Calculate valid start indices for each client
        # Map: global_index -> (client_id, local_start_index)
        # Optimized: cumsum of counts
        self.client_ids = []
        self.client_counts = []
        self.cumulative_counts = [0]
        
        total_samples = 0
        
        # Pre-calculate counts
        for client_id, series in client_data_dict.items():
            # How many sequences can we interpret from this series?
            # Length L. Valid starts: 0 to L - seq_len - horizon + 1
            n_samples = len(series) - sequence_length - forecast_horizon + 1
            if n_samples > 0:
                self.client_ids.append(client_id)
                self.client_counts.append(n_samples)
                total_samples += n_samples
                self.cumulative_counts.append(total_samples)
                
        self.total_samples = total_samples
        self.indices = np.arange(total_samples)
        self.cumulative_counts = np.array(self.cumulative_counts)
        
        print(f"Generator created: {len(self.client_ids)} clients, {total_samples:,} samples")
        
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.total_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get indices for this batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.total_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = []
        X_client_batch = []
        y_batch = []
        
        for global_idx in batch_indices:
            # Find which client this index belongs to using binary search on cumulative counts
            # searchsorted returns insertion point. 
            # cumulative is [0, 100, 250...]
            # if global_idx is 50, it falls in bucket 0 (insertion point 1)
            client_idx_idx = np.searchsorted(self.cumulative_counts, global_idx, side='right') - 1
            
            client_id = self.client_ids[client_idx_idx]
            # Local index within that client's valid sequences
            local_idx = global_idx - self.cumulative_counts[client_idx_idx]
            
            # Get data
            series = self.client_data_dict[client_id]
            
            # Slice: [local_idx : local_idx + seq_len + horizon]
            # We need +1 for slicing
            slice_end = local_idx + self.sequence_length + self.forecast_horizon
            segment = series.iloc[local_idx : slice_end]
            
            # Process this segment
            # 1. Split into input and target part logic
            segment_input = segment.iloc[:self.sequence_length]
            segment_target = segment.iloc[self.sequence_length:]
            
            # 2. Get scaler
            cid_enc = self.client_encoder.transform([client_id])[0]
            scaler = self.scalers[cid_enc]
            
            # 3. Add time features to input part
            input_time = add_time_features(segment_input) # (seq_len, 4)
            
            # 4. Scale consumption
            input_cons = scaler.transform(segment_input.values.reshape(-1, 1)) # (seq_len, 1)
            target_cons = scaler.transform(segment_target.values.reshape(-1, 1)) # (horizon, 1)
            
            # 5. Combine input
            input_combined = np.hstack([input_cons, input_time]) # (seq_len, 5)
            
            X_batch.append(input_combined)
            X_client_batch.append(cid_enc)
            y_batch.append(target_cons.flatten())
            
        return [np.array(X_batch), np.array(X_client_batch)], np.array(y_batch)
    
    def on_epoch_end(self):
        """Shuffle updates after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def prepare_multi_client_data_generators(trimmed_series, sequence_length=24, forecast_horizon=1, 
                                   test_size=0.2, validation_size=0.05, batch_size=256, console=None):
    """
    Prepare data generators for memory-efficient training.
    """
    if console is None:
        console = Console()
    
    console.print("   [cyan]Preparing data generators (Memory Efficient)...[/cyan]")
    
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
    
    # 3. Pre-process splits (Store raw Series for each split)
    # We do NOT create X_train arrays here. We just split the pandas Series objects.
    train_series = {}
    val_series = {}
    test_series = {}
    
    scalers = {}
    
    console.print("   [dim]Splitting data and fitting scalers...[/dim]")
    
    for client_id in valid_clients:
        series = trimmed_series[client_id]
        n = len(series)
        train_end_idx = int(n * (1 - test_size - validation_size))
        val_end_idx = int(n * (1 - test_size))
        
        # Split
        train_part = series.iloc[:train_end_idx]
        val_part = series.iloc[train_end_idx:val_end_idx] if val_end_idx > train_end_idx else None
        test_part = series.iloc[val_end_idx:] if val_end_idx < n else None
        
        # Fit Scaler on TRAIN only
        if len(train_part) > 10:
            scaler = MinMaxScaler()
            scaler.fit(train_part.values.reshape(-1, 1))
            
            cid_enc = client_encoder.transform([client_id])[0]
            scalers[cid_enc] = scaler
            
            # Store valid parts
            if len(train_part) >= sequence_length + forecast_horizon:
                train_series[client_id] = train_part
            
            if val_part is not None and len(val_part) >= sequence_length + forecast_horizon:
                val_series[client_id] = val_part
                
            if test_part is not None and len(test_part) >= sequence_length + forecast_horizon:
                test_series[client_id] = test_part
                
    console.print(f"   [green]Scalers fitted for {len(scalers)} clients[/green]")
    
    # 4. Create Generators
    train_gen = MultiClientDataGenerator(
        train_series, scalers, client_encoder,
        sequence_length=sequence_length, forecast_horizon=forecast_horizon,
        batch_size=batch_size, shuffle=True
    )
    
    val_gen = MultiClientDataGenerator(
        val_series, scalers, client_encoder,
        sequence_length=sequence_length, forecast_horizon=forecast_horizon,
        batch_size=batch_size, shuffle=False
    )
    
    test_gen = MultiClientDataGenerator(
        test_series, scalers, client_encoder,
        sequence_length=sequence_length, forecast_horizon=forecast_horizon,
        batch_size=batch_size, shuffle=False
    )
    
    console.print(f"   [green]Generators ready:[/green]")
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
                                   num_layers=1, input_dim=5):
    """
    Create multi-client LSTM model with client_id embedding.
    
    Args:
        sequence_length: Input sequence length
        n_clients: Number of unique clients
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
    # Shape: (sequence_length, input_dim) -> e.g., (24, 5)
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
    
    # LSTM layers - Improved architecture
    x = combined
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            name=f'lstm_{i+1}'
        )(x)
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
        name='multi_client_lstm'
    )
    
    # Compile model with improved optimizer settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
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
        input_dim=5  # Explicitly set to 5 for consumption + 4 time features
    )
    
    console.print(f"   [green]Model created with {model.count_params():,} parameters[/green]")
    
    # Train model
    console.print(f"\n[bold]Training model...[/bold]")
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
        patience=5,  # Reduced from 10 to 5 for faster training
        restore_best_weights=True,
        verbose=0  # Less verbose output
    )
    
    # ReduceLROnPlateau for faster convergence
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=0
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
    # Note: Generator yields ([X, ids], y)
    for i in range(len(test_gen)):
        (_, ids_batch), y_batch = test_gen[i]
        test_actual.extend(y_batch)
        test_client_ids.extend(ids_batch)
    
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
    metrics = {
        'mae': np.mean(np.abs(test_pred_inv - test_actual_inv)),
        'rmse': np.sqrt(np.mean((test_pred_inv - test_actual_inv)**2)),
        'mape': np.mean(np.abs((test_actual_inv - test_pred_inv) / (test_actual_inv + 1e-6))) * 100
    }            
        # data_dict['scaler'] is no longer used/available as a single object
        # but we need to return something compatible if needed
        # We will return the dict of scalers in the result map
        
        # Calculate metrics
        mae = np.mean(np.abs(test_pred_inv - test_actual_inv))
        rmse = np.sqrt(np.mean((test_pred_inv - test_actual_inv) ** 2))
        
        # Improved MAPE calculation
        threshold = np.abs(test_actual_inv).mean() * 0.01
        mask = np.abs(test_actual_inv) > threshold
        
        if mask.sum() > 0:
            mape = np.mean(np.abs((test_actual_inv[mask] - test_pred_inv[mask]) / test_actual_inv[mask])) * 100
        else:
            mape = np.mean(np.abs(test_actual_inv - test_pred_inv) / 
                          (np.abs(test_actual_inv) + np.abs(test_pred_inv) + 1e-8)) * 100
        
        mse = np.mean((test_pred_inv - test_actual_inv) ** 2)
        r2 = 1 - (np.sum((test_actual_inv - test_pred_inv) ** 2) / 
                 np.sum((test_actual_inv - np.mean(test_actual_inv)) ** 2))
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'mse': float(mse),
            'r2': float(r2)
        }
        
        console.print(f"\n[bold]Test Set Metrics:[/bold]")
        console.print(f"   [green]MAE: {metrics['mae']:.2f} kW[/green]")
        console.print(f"   [green]RMSE: {metrics['rmse']:.2f} kW[/green]")
        console.print(f"   [green]MAPE: {metrics['mape']:.2f}%[/green]")
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
        'batch_size': 256,  # Reduced for better gradient updates
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
