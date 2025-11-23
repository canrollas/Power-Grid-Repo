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
    
    for i in range(len(series_scaled) - sequence_length - forecast_horizon + 1):
        X.append(series_scaled[i:i + sequence_length])
        X_client.append(client_id_encoded)
        y.append(series_scaled[i + sequence_length:i + sequence_length + forecast_horizon])
    
    return X, X_client, y


def prepare_multi_client_data(trimmed_series, sequence_length=24, forecast_horizon=1, 
                              test_size=0.2, validation_size=0.05, console=None):
    """
    Prepare multi-client data for LSTM training with TRUE time-based split.
    
    Her client için ayrı ayrı zaman bazlı split yapar:
    - Her client'ın ilk %(1-test_size-validation_size) zamanı → train
    - Her client'ın ortadaki %validation_size zamanı → validation
    - Her client'ın son %test_size zamanı → test
    - Sequence'ler client sınırı içinde kalır
    - Scaler sadece train verisinden fit edilir (no data leakage)
    
    Args:
        trimmed_series: dict[str, pd.Series] - Her client için trimmed seri (farklı uzunlukta zaman index'i)
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of steps ahead to predict
        test_size: Proportion of data for testing (default: 0.2 for 20%)
        validation_size: Proportion of data for validation (default: 0.05 for 5%)
        console: Rich Console object for colored output
    
    Returns:
        dict: Contains scaled data, scaler, train/val/test splits, and client encoder
    """
    if console is None:
        console = Console()
    
    console.print("   [cyan]Preparing multi-client dataset with time-based split...[/cyan]")
    
    # Adım 1: Client ID encoder'ı oluştur
    valid_clients = []
    for client_id, client_series in trimmed_series.items():
        if len(client_series) >= sequence_length + forecast_horizon + 10:
            valid_clients.append(client_id)
    
    if len(valid_clients) == 0:
        console.print("   [red]Error: No valid data found[/red]")
        return None
    
    client_encoder = LabelEncoder()
    client_encoder.fit(valid_clients)
    
    console.print(f"   [green]Number of valid clients: {len(valid_clients)}[/green]")
    
    # Adım 2: Sadece train verisinden scaler fit et
    train_values = []
    
    for client_id in valid_clients:
        client_series = trimmed_series[client_id]
        n = len(client_series)
        train_end_idx = int(n * (1 - test_size - validation_size))  # örn. 0.75
        
        if train_end_idx < sequence_length + forecast_horizon:
            continue  # Train kısmı çok kısa, skip
        
        train_part = client_series.iloc[:train_end_idx]
        train_values.append(train_part.values)
    
    if len(train_values) == 0:
        console.print("   [red]Error: No valid train data found[/red]")
        return None
    
    train_values = np.concatenate(train_values).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(train_values)
    
    console.print(f"   [green]Scaler fitted on {len(train_values):,} train data points[/green]")
    
    # Adım 3: Her client için ayrı ayrı sequence üret
    X_train, y_train, c_train = [], [], []
    X_val, y_val, c_val = [], [], []
    X_test, y_test, c_test = [], [], []
    only_train_clients = 0
    full_split_clients = 0
    
    for client_id in valid_clients:
        client_series = trimmed_series[client_id]
        n = len(client_series)
        train_end_idx = int(n * (1 - test_size - validation_size))  # örn. 0.75
        val_end_idx = int(n * (1 - test_size))  # örn. 0.95
        
        if train_end_idx < sequence_length + forecast_horizon:
            continue  # Train kısmı çok kısa, skip
        
        train_part = client_series.iloc[:train_end_idx]
        val_part = client_series.iloc[train_end_idx:val_end_idx] if val_end_idx > train_end_idx else None
        test_part = client_series.iloc[val_end_idx:] if val_end_idx < n else None
        
        # Client ID'yi encode et
        cid_enc = client_encoder.transform([client_id])[0]
        
        # Scale train kısmı
        train_scaled = scaler.transform(train_part.values.reshape(-1, 1)).flatten()
        Xt, Ct, yt = make_sequences_1d(train_scaled, cid_enc, sequence_length, forecast_horizon)
        X_train.extend(Xt)
        c_train.extend(Ct)
        y_train.extend(yt)
        
        # Validation kısmı yeterli uzunlukta mı?
        if val_part is not None and len(val_part) >= sequence_length + forecast_horizon:
            val_scaled = scaler.transform(val_part.values.reshape(-1, 1)).flatten()
            Xv, Cv, yv = make_sequences_1d(val_scaled, cid_enc, sequence_length, forecast_horizon)
            X_val.extend(Xv)
            c_val.extend(Cv)
            y_val.extend(yv)
        else:
            # Validation kısmı çok kısa, sadece train'e ekle (zaten eklendi)
            pass
        
        # Test kısmı yeterli uzunlukta mı?
        if test_part is not None and len(test_part) >= sequence_length + forecast_horizon:
            test_scaled = scaler.transform(test_part.values.reshape(-1, 1)).flatten()
            Xtst, Ctst, ytst = make_sequences_1d(test_scaled, cid_enc, sequence_length, forecast_horizon)
            X_test.extend(Xtst)
            c_test.extend(Ctst)
            y_test.extend(ytst)
            full_split_clients += 1
        else:
            # Test kısmı çok kısa, sadece train'e ekle (zaten eklendi)
            only_train_clients += 1
    
    if len(X_train) == 0:
        console.print("   [red]Error: No train sequences created[/red]")
        return None
    
    # Numpy array'lere dönüştür
    X_train = np.array(X_train)[..., np.newaxis]  # (N, seq_len, 1)
    X_val = np.array(X_val)[..., np.newaxis] if len(X_val) > 0 else np.array([]).reshape(0, sequence_length, 1)
    X_test = np.array(X_test)[..., np.newaxis] if len(X_test) > 0 else np.array([]).reshape(0, sequence_length, 1)
    y_train = np.array(y_train)
    y_val = np.array(y_val) if len(y_val) > 0 else np.array([]).reshape(0, forecast_horizon)
    y_test = np.array(y_test) if len(y_test) > 0 else np.array([]).reshape(0, forecast_horizon)
    c_train = np.array(c_train)
    c_val = np.array(c_val) if len(c_val) > 0 else np.array([])
    c_test = np.array(c_test) if len(c_test) > 0 else np.array([])
    
    console.print(f"   [green]Created {len(X_train):,} train sequences[/green]")
    if len(X_val) > 0:
        console.print(f"   [green]Created {len(X_val):,} validation sequences[/green]")
    if len(X_test) > 0:
        console.print(f"   [green]Created {len(X_test):,} test sequences[/green]")
    else:
        console.print(f"   [yellow]Warning: No test sequences created (test parts too short)[/yellow]")
    
    if only_train_clients > 0:
        console.print(f"   [dim]Only-train clients: {only_train_clients} (test parts too short)[/dim]")
    if full_split_clients > 0:
        console.print(f"   [dim]Full-split clients: {full_split_clients} (train/val/test)[/dim]")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_client_train': c_train,
        'X_client_val': c_val,
        'X_client_test': c_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
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
        
        if self.total_batches:
            self.progress.update(
                self.batch_task,
                advance=1,
                description=f"  [dim]Batch {self.current_batch}/{self.total_batches} - loss: {loss:.6e}, mae: {mae:.6e}[/dim]"
            )
        else:
            self.progress.update(
                self.batch_task,
                description=f"  [dim]Batch {self.current_batch} - loss: {loss:.6e}, mae: {mae:.6e}[/dim]"
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
        
        self.console.print(
            f"   [green]✓[/green] Epoch {self.current_epoch}/{self.total_epochs} - "
            f"loss: {loss:.6e}, mae: {mae:.6e}, "
            f"val_loss: {val_loss:.6e}, val_mae: {val_mae:.6e}, "
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
                                   num_layers=1):
    """
    Create multi-client LSTM model with client_id embedding.
    
    Args:
        sequence_length: Input sequence length
        n_clients: Number of unique clients
        embedding_dim: Dimension of client embedding
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        num_layers: Number of LSTM layers
    
    Returns:
        Compiled Keras model
    """
    # Input 1: Time series sequence
    sequence_input = tf.keras.Input(shape=(sequence_length, 1), name='consumption_sequence')
    
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
    
    # LSTM layers
    x = combined
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            name=f'lstm_{i+1}'
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Dense output layer (cast to float32 for mixed precision)
    output = tf.keras.layers.Dense(1, name='consumption_prediction', dtype='float32')(x)
    
    # Create model
    model = tf.keras.Model(
        inputs=[sequence_input, client_input],
        outputs=output,
        name='multi_client_lstm'
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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
    
    # Prepare data
    data_dict = prepare_multi_client_data(
        trimmed_series,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        test_size=0.2,  # 20% test
        validation_size=validation_size,  # 5% validation
        console=console
    )
    
    if data_dict is None:
        return None
    
    n_clients = data_dict['n_clients']
    
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
        num_layers=num_layers
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
    has_validation = len(data_dict['X_val']) > 0
    
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
    train_size = len(data_dict['X_train'])
    steps_per_epoch = (train_size + batch_size - 1) // batch_size  # Ceiling division
    rich_progress = RichProgressCallback(total_epochs=epochs, console=console)
    rich_progress.total_batches = steps_per_epoch
    
    try:
        # Validation seti varsa manuel olarak geç, yoksa validation_split kullanma
        if has_validation:
            history = model.fit(
                [data_dict['X_train'], data_dict['X_client_train']],
                data_dict['y_train'],
                validation_data=([data_dict['X_val'], data_dict['X_client_val']], data_dict['y_val']),
                epochs=epochs,
                batch_size=batch_size,
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
                [data_dict['X_train'], data_dict['X_client_train']],
                data_dict['y_train'],
                epochs=epochs,
                batch_size=batch_size,
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
    
    # Evaluate on test set
    if len(data_dict['X_test']) == 0:
        console.print(f"\n[yellow]Warning: Test set is empty, skipping evaluation[/yellow]")
        console.print(f"   [dim]This can happen if test parts of clients are too short[/dim]")
        metrics = {
            'mae': None,
            'rmse': None,
            'mape': None,
            'mse': None,
            'r2': None
        }
        test_pred_inv = None
        test_actual_inv = None
    else:
        console.print(f"\n[bold]Evaluating on test set...[/bold]")
        test_predictions = model.predict(
            [data_dict['X_test'], data_dict['X_client_test']],
            batch_size=batch_size,
            verbose=0
        )
        test_actual = data_dict['y_test']
        
        # Inverse transform
        test_pred_inv = data_dict['scaler'].inverse_transform(test_predictions)
        test_actual_inv = data_dict['scaler'].inverse_transform(test_actual)
        
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
        'scaler': data_dict['scaler'],
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
    
    # Default parameters (optimized for speed)
    default_params = {
        'sequence_length': 24,
        'forecast_horizon': 1,
        'embedding_dim': 8,  # Reduced from 16
        'lstm_units': 32,  # Reduced from 64
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'num_layers': 1,  # Reduced from 2
        'batch_size': 512,  # Increased from 128 (4x faster)
        'epochs': 20,  # Reduced from 50
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
