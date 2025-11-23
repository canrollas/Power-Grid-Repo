#!/usr/bin/env python3
"""
Error Handler Module

Handles errors gracefully with automatic recovery strategies.
Especially handles CUDA OOM errors by reducing batch size and retrying.
"""

import traceback
from typing import Dict, Any, Callable, Optional, Tuple
import tensorflow as tf
from rich.console import Console


class ErrorHandler:
    """
    Handles errors with automatic recovery strategies.
    """
    
    def __init__(self, tracker=None, console=None):
        """
        Initialize error handler.
        
        Args:
            tracker: ExperimentTracker instance for logging
            console: Rich Console for output
        """
        self.tracker = tracker
        self.console = console or Console()
        self.recovery_strategies = {
            'CUDA_OOM': self._recover_cuda_oom,
            'ResourceExhaustedError': self._recover_cuda_oom,
            'OutOfMemoryError': self._recover_cuda_oom,
            'MemoryError': self._recover_memory_error,
        }
    
    def handle_error(self, error: Exception, context: Dict = None, 
                    retry_func: Callable = None, max_retries: int = 3) -> Tuple[Any, bool]:
        """
        Handle an error with automatic recovery.
        
        Args:
            error: Exception that occurred
            context: Context information (client_id, step, etc.)
            retry_func: Function to retry after recovery
            max_retries: Maximum number of retry attempts
        
        Returns:
            Tuple of (result, success)
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log error
        if self.tracker:
            self.tracker.log_error(
                error_type=error_type,
                error_message=error_message,
                context=context or {}
            )
        
        # Check if we have a recovery strategy
        recovery_strategy = None
        for key, strategy in self.recovery_strategies.items():
            if key in error_type or key in error_message:
                recovery_strategy = strategy
                break
        
        if recovery_strategy and retry_func:
            # Try to recover
            recovery_action = recovery_strategy(error, context)
            
            if recovery_action and recovery_action.get('success', False):
                # Retry the operation
                for attempt in range(max_retries):
                    try:
                        if self.console:
                            self.console.print(f"   [yellow]Retry attempt {attempt + 1}/{max_retries}...[/yellow]")
                        
                        result = retry_func(**recovery_action.get('params', {}))
                        
                        # Log successful recovery
                        if self.tracker:
                            self.tracker.log_error(
                                error_type=error_type,
                                error_message=error_message,
                                context=context or {},
                                recovered=True,
                                recovery_action=recovery_action.get('action', '')
                            )
                        
                        if self.console:
                            self.console.print(f"   [green]Recovery successful![/green]")
                        
                        return result, True
                    except Exception as retry_error:
                        if attempt < max_retries - 1:
                            # Try different recovery strategy or adjust parameters
                            recovery_action = recovery_strategy(retry_error, context, 
                                                              previous_action=recovery_action)
                            if not recovery_action or not recovery_action.get('success', False):
                                break
                        else:
                            # Final attempt failed
                            if self.console:
                                self.console.print(f"   [red]Recovery failed after {max_retries} attempts[/red]")
                            if self.tracker:
                                self.tracker.log_error(
                                    error_type=type(retry_error).__name__,
                                    error_message=str(retry_error),
                                    context=context or {},
                                    recovered=False,
                                    recovery_action="Max retries exceeded"
                                )
                            return None, False
        
        # No recovery strategy or recovery failed
        if self.console:
            self.console.print(f"   [red]Error: {error_type}: {error_message}[/red]")
        
        return None, False
    
    def _recover_cuda_oom(self, error: Exception, context: Dict = None, 
                         previous_action: Dict = None) -> Dict:
        """
        Recover from CUDA Out of Memory error.
        
        Strategy:
        1. Clear TensorFlow session
        2. Reduce batch size
        3. Enable memory growth if not already enabled
        
        Args:
            error: The OOM error
            context: Context information
            previous_action: Previous recovery action (for iterative reduction)
        
        Returns:
            Recovery action dictionary
        """
        # Get current batch size from context or previous action
        current_batch_size = context.get('batch_size', 128) if context else 128
        if previous_action:
            current_batch_size = previous_action.get('params', {}).get('batch_size', current_batch_size)
        
        # Reduce batch size (halve it, minimum 1)
        new_batch_size = max(1, current_batch_size // 2)
        
        if new_batch_size == current_batch_size:
            # Can't reduce further
            return {'success': False, 'action': 'Cannot reduce batch size further'}
        
        # Clear TensorFlow session
        try:
            tf.keras.backend.clear_session()
        except:
            pass
        
        # Try to enable memory growth
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
        
        action = f"Reduced batch size from {current_batch_size} to {new_batch_size}, cleared TF session"
        
        if self.console:
            self.console.print(f"   [yellow]CUDA OOM detected. {action}[/yellow]")
        
        return {
            'success': True,
            'action': action,
            'params': {
                'batch_size': new_batch_size
            }
        }
    
    def _recover_memory_error(self, error: Exception, context: Dict = None, 
                             previous_action: Dict = None) -> Dict:
        """
        Recover from general memory error.
        
        Args:
            error: The memory error
            context: Context information
            previous_action: Previous recovery action
        
        Returns:
            Recovery action dictionary
        """
        # Similar to CUDA OOM but for CPU memory
        current_batch_size = context.get('batch_size', 128) if context else 128
        if previous_action:
            current_batch_size = previous_action.get('params', {}).get('batch_size', current_batch_size)
        
        new_batch_size = max(1, current_batch_size // 2)
        
        if new_batch_size == current_batch_size:
            return {'success': False, 'action': 'Cannot reduce batch size further'}
        
        # Clear session
        try:
            tf.keras.backend.clear_session()
        except:
            pass
        
        action = f"Reduced batch size from {current_batch_size} to {new_batch_size} due to memory error"
        
        if self.console:
            self.console.print(f"   [yellow]Memory error detected. {action}[/yellow]")
        
        return {
            'success': True,
            'action': action,
            'params': {
                'batch_size': new_batch_size
            }
        }
    
    def safe_execute(self, func: Callable, context: Dict = None, 
                    max_retries: int = 3, **kwargs) -> Tuple[Any, bool]:
        """
        Safely execute a function with error handling and recovery.
        
        Args:
            func: Function to execute
            context: Context information
            max_retries: Maximum retry attempts
            **kwargs: Arguments to pass to function
        
        Returns:
            Tuple of (result, success)
        """
        def retry_wrapper(**retry_kwargs):
            """Wrapper for retry with updated parameters."""
            updated_kwargs = {**kwargs, **retry_kwargs}
            return func(**updated_kwargs)
        
        try:
            result = func(**kwargs)
            return result, True
        except Exception as e:
            return self.handle_error(e, context, retry_func=retry_wrapper, max_retries=max_retries)

