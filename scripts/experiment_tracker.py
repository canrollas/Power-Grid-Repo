#!/usr/bin/env python3
"""
Experiment Tracking Module

Tracks all metrics, hyperparameters, trials, errors, and logs throughout the experiment.
Saves everything to JSON files for later analysis and reporting.
"""

import json
import os
from datetime import datetime
from typing import Dict,  Any
import numpy as np


class ExperimentTracker:
    """
    Comprehensive experiment tracking system.
    Tracks metrics, hyperparameters, trials, errors, and execution logs.
    """
    
    def __init__(self, experiment_name: str = None, base_dir: str = "data/processed/experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name for this experiment (auto-generated if None)
            base_dir: Base directory for saving experiment data
        """
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize tracking dictionaries
        self.metrics = {}  # All metrics by client
        self.hyperparameters = {}  # All hyperparameters tried
        self.trials = []  # All Optuna trials
        self.errors = []  # All errors encountered
        self.execution_log = []  # Execution steps and logs
        self.client_results = {}  # Results per client
        self.system_info = {}  # System information
        self.plots = []  # Plot file paths
        
        # Start tracking
        self.log_step("EXPERIMENT_START", f"Experiment '{experiment_name}' started")
        self._save_system_info()
    
    def log_step(self, step_name: str, message: str, data: Dict = None):
        """
        Log an execution step.
        
        Args:
            step_name: Name of the step
            message: Log message
            data: Optional additional data
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'message': message,
            'data': data or {}
        }
        self.execution_log.append(log_entry)
        self._save_logs()
    
    def log_error(self, error_type: str, error_message: str, context: Dict = None, 
                  recovered: bool = False, recovery_action: str = None):
        """
        Log an error with context.
        
        Args:
            error_type: Type of error (e.g., 'CUDA_OOM', 'ValueError')
            error_message: Error message
            context: Additional context (client_id, step, etc.)
            recovered: Whether the error was recovered from
            recovery_action: Action taken to recover
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {},
            'recovered': recovered,
            'recovery_action': recovery_action
        }
        self.errors.append(error_entry)
        self.log_step("ERROR", f"{error_type}: {error_message}", error_entry)
        self._save_errors()
    
    def log_trial(self, client_id: str, trial_number: int, params: Dict, 
                  value: float, state: str = "COMPLETE"):
        """
        Log an Optuna trial.
        
        Args:
            client_id: Client identifier
            trial_number: Trial number
            params: Hyperparameters tried
            value: Objective value (loss)
            state: Trial state (COMPLETE, PRUNED, FAIL)
        """
        trial_entry = {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'trial_number': trial_number,
            'params': params,
            'value': value,
            'state': state
        }
        self.trials.append(trial_entry)
        self._save_trials()
    
    def log_client_result(self, client_id: str, metrics: Dict, best_params: Dict, 
                         best_trial_value: float, model_info: Dict = None):
        """
        Log final results for a client.
        
        Args:
            client_id: Client identifier
            metrics: Evaluation metrics (MAE, RMSE, MAPE, etc.)
            best_params: Best hyperparameters found
            best_trial_value: Best trial objective value
            model_info: Additional model information
        """
        result = {
            'client_id': client_id,
            'metrics': metrics,
            'best_params': best_params,
            'best_trial_value': best_trial_value,
            'model_info': model_info or {},
            'timestamp': datetime.now().isoformat()
        }
        self.client_results[client_id] = result
        self.metrics[client_id] = metrics
        self._save_results()
    
    def log_hyperparameter(self, client_id: str, param_name: str, param_value: Any):
        """
        Log a hyperparameter value.
        
        Args:
            client_id: Client identifier
            param_name: Parameter name
            param_value: Parameter value
        """
        if client_id not in self.hyperparameters:
            self.hyperparameters[client_id] = {}
        self.hyperparameters[client_id][param_name] = param_value
        self._save_hyperparameters()
    
    def add_plot(self, plot_type: str, plot_path: str, description: str = None):
        """
        Register a plot file.
        
        Args:
            plot_type: Type of plot (e.g., 'consumption', 'categorization', 'prediction')
            plot_path: Path to plot file
            description: Optional description
        """
        plot_entry = {
            'type': plot_type,
            'path': plot_path,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        self.plots.append(plot_entry)
        self._save_plots()
    
    def get_summary(self) -> Dict:
        """
        Get experiment summary.
        
        Returns:
            Dictionary with experiment summary
        """
        total_clients = len(self.client_results)
        successful_clients = sum(1 for r in self.client_results.values() 
                                if 'metrics' in r and r['metrics'])
        failed_clients = total_clients - successful_clients
        
        # Aggregate metrics
        if self.metrics:
            mae_values = [m.get('mae', np.nan) for m in self.metrics.values() if m]
            rmse_values = [m.get('rmse', np.nan) for m in self.metrics.values() if m]
            mape_values = [m.get('mape', np.nan) for m in self.metrics.values() if m]
            
            metrics_summary = {
                'mae': {
                    'mean': float(np.nanmean(mae_values)) if mae_values else None,
                    'median': float(np.nanmedian(mae_values)) if mae_values else None,
                    'std': float(np.nanstd(mae_values)) if mae_values else None,
                    'min': float(np.nanmin(mae_values)) if mae_values else None,
                    'max': float(np.nanmax(mae_values)) if mae_values else None
                },
                'rmse': {
                    'mean': float(np.nanmean(rmse_values)) if rmse_values else None,
                    'median': float(np.nanmedian(rmse_values)) if rmse_values else None,
                    'std': float(np.nanstd(rmse_values)) if rmse_values else None,
                    'min': float(np.nanmin(rmse_values)) if rmse_values else None,
                    'max': float(np.nanmax(rmse_values)) if rmse_values else None
                },
                'mape': {
                    'mean': float(np.nanmean(mape_values)) if mape_values else None,
                    'median': float(np.nanmedian(mape_values)) if mape_values else None,
                    'std': float(np.nanstd(mape_values)) if mape_values else None,
                    'min': float(np.nanmin(mape_values)) if mape_values else None,
                    'max': float(np.nanmax(mape_values)) if mape_values else None
                }
            }
        else:
            metrics_summary = {}
        
        return {
            'experiment_name': self.experiment_name,
            'experiment_dir': self.experiment_dir,
            'total_clients': total_clients,
            'successful_clients': successful_clients,
            'failed_clients': failed_clients,
            'total_trials': len(self.trials),
            'total_errors': len(self.errors),
            'recovered_errors': sum(1 for e in self.errors if e.get('recovered', False)),
            'metrics_summary': metrics_summary,
            'start_time': self.execution_log[0]['timestamp'] if self.execution_log else None,
            'end_time': datetime.now().isoformat()
        }
    
    def finalize(self):
        """Finalize experiment and save all data."""
        self.log_step("EXPERIMENT_END", f"Experiment '{self.experiment_name}' completed")
        summary = self.get_summary()
        self._save_summary(summary)
        return summary
    
    def _save_system_info(self):
        """Save system information."""
        import platform
        import sys
        
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        except:
            tf_version = "N/A"
            gpu_available = False
        
        self.system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'tensorflow_version': tf_version,
            'gpu_available': gpu_available,
            'experiment_start': datetime.now().isoformat()
        }
        self._save_file('system_info.json', self.system_info)
    
    def _save_logs(self):
        """Save execution logs."""
        self._save_file('execution_log.json', self.execution_log)
    
    def _save_errors(self):
        """Save error logs."""
        self._save_file('errors.json', self.errors)
    
    def _save_trials(self):
        """Save trial data."""
        self._save_file('trials.json', self.trials)
    
    def _save_results(self):
        """Save client results."""
        self._save_file('client_results.json', self.client_results)
    
    def _save_hyperparameters(self):
        """Save hyperparameters."""
        self._save_file('hyperparameters.json', self.hyperparameters)
    
    def _save_plots(self):
        """Save plot registry."""
        self._save_file('plots.json', self.plots)
    
    def _save_summary(self, summary: Dict):
        """Save experiment summary."""
        self._save_file('summary.json', summary)
    
    def _save_file(self, filename: str, data: Any):
        """Save data to JSON file."""
        filepath = os.path.join(self.experiment_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

