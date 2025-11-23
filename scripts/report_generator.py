#!/usr/bin/env python3
"""
Report Generator Module

Generates comprehensive HTML reports with plots, metrics, heatmaps, and analysis.
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ReportGenerator:
    """
    Generates comprehensive HTML reports from experiment data.
    """
    
    def __init__(self, tracker, output_path: str = None):
        """
        Initialize report generator.
        
        Args:
            tracker: ExperimentTracker instance
            output_path: Path for output HTML file (auto-generated if None)
        """
        self.tracker = tracker
        if output_path is None:
            output_path = os.path.join(tracker.experiment_dir, 'report.html')
        self.output_path = output_path
        self.plots_dir = os.path.join(tracker.experiment_dir, 'report_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def generate_report(self) -> str:
        """
        Generate comprehensive HTML report.
        
        Returns:
            Path to generated report
        """
        summary = self.tracker.get_summary()
        client_results = self.tracker.client_results
        trials = self.tracker.trials
        errors = self.tracker.errors
        execution_log = self.tracker.execution_log
        
        # Generate plots
        plots_html = self._generate_plots(client_results, trials, errors)
        
        # Generate HTML
        html_content = self._generate_html(
            summary=summary,
            client_results=client_results,
            trials=trials,
            errors=errors,
            execution_log=execution_log,
            plots_html=plots_html
        )
        
        # Save report
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return self.output_path
    
    def _generate_plots(self, client_results: Dict, trials: List, errors: List) -> str:
        """Generate all plots and return HTML for embedding."""
        plots_html = []
        
        # 1. Metrics distribution plots
        plots_html.append(self._plot_metrics_distribution(client_results))
        
        # 2. Hyperparameter heatmap
        plots_html.append(self._plot_hyperparameter_heatmap(trials))
        
        # 3. Trial convergence plots
        plots_html.append(self._plot_trial_convergence(trials))
        
        # 4. Error analysis
        if errors:
            plots_html.append(self._plot_error_analysis(errors))
        
        # 5. Client performance comparison
        plots_html.append(self._plot_client_performance(client_results))
        
        # 6. Hyperparameter importance
        plots_html.append(self._plot_hyperparameter_importance(trials))
        
        return '\n'.join(plots_html)
    
    def _plot_metrics_distribution(self, client_results: Dict) -> str:
        """Plot distribution of metrics."""
        if not client_results:
            return ""
        
        metrics_data = {
            'MAE': [r['metrics'].get('mae', np.nan) for r in client_results.values() if r.get('metrics')],
            'RMSE': [r['metrics'].get('rmse', np.nan) for r in client_results.values() if r.get('metrics')],
            'MAPE': [r['metrics'].get('mape', np.nan) for r in client_results.values() if r.get('metrics')]
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            values = [v for v in values if not np.isnan(v)]
            if values:
                axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(metric_name)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].axvline(np.mean(values), color='red', linestyle='--', 
                                label=f'Mean: {np.mean(values):.2f}')
                axes[idx].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'metrics_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{plot_path}" alt="Metrics Distribution" style="max-width:100%;">'
    
    def _plot_hyperparameter_heatmap(self, trials: List) -> str:
        """Plot hyperparameter correlation heatmap."""
        if not trials:
            return ""
        
        # Extract hyperparameters and values
        trial_data = []
        for trial in trials:
            if trial.get('params') and trial.get('value') is not None:
                row = trial['params'].copy()
                row['objective_value'] = trial['value']
                trial_data.append(row)
        
        if not trial_data:
            return ""
        
        df = pd.DataFrame(trial_data)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'objective_value' in numeric_cols:
            numeric_cols.remove('objective_value')
        
        if len(numeric_cols) < 2:
            return ""
        
        # Calculate correlation
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Hyperparameter Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, 'hyperparameter_heatmap.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{plot_path}" alt="Hyperparameter Heatmap" style="max-width:100%;">'
    
    def _plot_trial_convergence(self, trials: List) -> str:
        """Plot trial convergence over time."""
        if not trials:
            return ""
        
        # Group by client
        client_trials = {}
        for trial in trials:
            client_id = trial.get('client_id', 'unknown')
            if client_id not in client_trials:
                client_trials[client_id] = []
            if trial.get('value') is not None:
                client_trials[client_id].append({
                    'trial': trial.get('trial_number', 0),
                    'value': trial['value']
                })
        
        if not client_trials:
            return ""
        
        # Plot first few clients
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (client_id, trials_list) in enumerate(list(client_trials.items())[:4]):
            if idx >= len(axes):
                break
            
            trials_list = sorted(trials_list, key=lambda x: x['trial'])
            trials_nums = [t['trial'] for t in trials_list]
            values = [t['value'] for t in trials_list]
            
            axes[idx].plot(trials_nums, values, marker='o', linewidth=2, markersize=4)
            axes[idx].set_title(f'Client {client_id} - Trial Convergence', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Trial Number')
            axes[idx].set_ylabel('Objective Value (Loss)')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(list(client_trials.items())[:4]), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'trial_convergence.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{plot_path}" alt="Trial Convergence" style="max-width:100%;">'
    
    def _plot_error_analysis(self, errors: List) -> str:
        """Plot error analysis."""
        if not errors:
            return ""
        
        # Count errors by type
        error_counts = {}
        for error in errors:
            error_type = error.get('error_type', 'Unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Plot pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        if error_counts:
            ax1.pie(error_counts.values(), labels=error_counts.keys(), autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax1.set_title('Error Types Distribution', fontsize=12, fontweight='bold')
        
        # Recovery success rate
        recovered = sum(1 for e in errors if e.get('recovered', False))
        not_recovered = len(errors) - recovered
        
        if recovered + not_recovered > 0:
            ax2.bar(['Recovered', 'Not Recovered'], [recovered, not_recovered], 
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
            ax2.set_title('Error Recovery Rate', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Count')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'error_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{plot_path}" alt="Error Analysis" style="max-width:100%;">'
    
    def _plot_client_performance(self, client_results: Dict) -> str:
        """Plot client performance comparison."""
        if not client_results:
            return ""
        
        # Get top and bottom performers
        clients_with_metrics = [(cid, r['metrics']) for cid, r in client_results.items() 
                               if r.get('metrics')]
        
        if not clients_with_metrics:
            return ""
        
        # Sort by MAE
        clients_with_metrics.sort(key=lambda x: x[1].get('mae', np.inf))
        
        # Top 10 and bottom 10
        top_10 = clients_with_metrics[:10]
        bottom_10 = clients_with_metrics[-10:] if len(clients_with_metrics) >= 10 else clients_with_metrics
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top 10
        top_clients = [c[0] for c in top_10]
        top_mae = [c[1].get('mae', 0) for c in top_10]
        
        ax1.barh(range(len(top_clients)), top_mae, color='green', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(top_clients)))
        ax1.set_yticklabels([c[:15] + '...' if len(c) > 15 else c for c in top_clients], fontsize=8)
        ax1.set_xlabel('MAE (kW)')
        ax1.set_title('Top 10 Best Performing Clients', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Bottom 10
        bottom_clients = [c[0] for c in bottom_10]
        bottom_mae = [c[1].get('mae', 0) for c in bottom_10]
        
        ax2.barh(range(len(bottom_clients)), bottom_mae, color='red', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(bottom_clients)))
        ax2.set_yticklabels([c[:15] + '...' if len(c) > 15 else c for c in bottom_clients], fontsize=8)
        ax2.set_xlabel('MAE (kW)')
        ax2.set_title('Bottom 10 Worst Performing Clients', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'client_performance.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{plot_path}" alt="Client Performance" style="max-width:100%;">'
    
    def _plot_hyperparameter_importance(self, trials: List) -> str:
        """Plot hyperparameter importance analysis."""
        if not trials:
            return ""
        
        # Extract hyperparameters and objective values
        param_data = {}
        for trial in trials:
            if trial.get('params') and trial.get('value') is not None:
                for param_name, param_value in trial['params'].items():
                    if isinstance(param_value, (int, float)):
                        if param_name not in param_data:
                            param_data[param_name] = []
                        param_data[param_name].append((param_value, trial['value']))
        
        if not param_data:
            return ""
        
        # Calculate correlation with objective for each parameter
        param_importance = {}
        for param_name, values in param_data.items():
            if len(values) > 1:
                param_vals = [v[0] for v in values]
                obj_vals = [v[1] for v in values]
                correlation = np.abs(np.corrcoef(param_vals, obj_vals)[0, 1])
                if not np.isnan(correlation):
                    param_importance[param_name] = correlation
        
        if not param_importance:
            return ""
        
        # Sort by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        param_names = [p[0] for p in sorted_params]
        importances = [p[1] for p in sorted_params]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(param_names)), importances, color='steelblue', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(param_names)), param_names)
        plt.xlabel('Absolute Correlation with Objective Value', fontsize=11, fontweight='bold')
        plt.title('Hyperparameter Importance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, 'hyperparameter_importance.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{plot_path}" alt="Hyperparameter Importance" style="max-width:100%;">'
    
    def _generate_html(self, summary: Dict, client_results: Dict, trials: List, 
                      errors: List, execution_log: List, plots_html: str) -> str:
        """Generate HTML content."""
        
        # Metrics table
        metrics_table = self._generate_metrics_table(client_results)
        
        # Errors table
        errors_table = self._generate_errors_table(errors)
        
        # Trials summary
        trials_summary = self._generate_trials_summary(trials)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - {summary['experiment_name']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
            background-color: #fafafa;
            padding: 20px;
            border-radius: 8px;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .error-recovered {{
            color: green;
            font-weight: bold;
        }}
        .error-not-recovered {{
            color: red;
            font-weight: bold;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Report: {summary['experiment_name']}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Clients</h3>
                <div class="value">{summary['total_clients']}</div>
            </div>
            <div class="summary-card">
                <h3>Successful</h3>
                <div class="value">{summary['successful_clients']}</div>
            </div>
            <div class="summary-card">
                <h3>Failed</h3>
                <div class="value">{summary['failed_clients']}</div>
            </div>
            <div class="summary-card">
                <h3>Total Trials</h3>
                <div class="value">{summary['total_trials']}</div>
            </div>
            <div class="summary-card">
                <h3>Errors</h3>
                <div class="value">{summary['total_errors']}</div>
            </div>
            <div class="summary-card">
                <h3>Recovered</h3>
                <div class="value">{summary['recovered_errors']}</div>
            </div>
        </div>
        
        <h2>Metrics Summary</h2>
        {self._generate_metrics_summary_html(summary.get('metrics_summary', {}))}
        
        <h2>Visualizations</h2>
        {plots_html}
        
        <h2>Client Results</h2>
        {metrics_table}
        
        <h2>Hyperparameter Trials</h2>
        {trials_summary}
        
        <h2>Errors & Recovery</h2>
        {errors_table}
        
        <h2>Execution Log</h2>
        <div style="max-height: 400px; overflow-y: auto; background-color: #fafafa; padding: 15px; border-radius: 5px;">
            <pre style="font-family: 'Courier New', monospace; font-size: 11px; margin: 0;">{self._format_execution_log(execution_log)}</pre>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_metrics_summary_html(self, metrics_summary: Dict) -> str:
        """Generate metrics summary HTML."""
        if not metrics_summary:
            return "<p>No metrics available.</p>"
        
        html = "<table><tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr>"
        
        for metric_name, stats in metrics_summary.items():
            html += f"""
            <tr>
                <td><strong>{metric_name.upper()}</strong></td>
                <td>{stats.get('mean', 'N/A'):.2f if stats.get('mean') is not None else 'N/A'}</td>
                <td>{stats.get('median', 'N/A'):.2f if stats.get('median') is not None else 'N/A'}</td>
                <td>{stats.get('std', 'N/A'):.2f if stats.get('std') is not None else 'N/A'}</td>
                <td>{stats.get('min', 'N/A'):.2f if stats.get('min') is not None else 'N/A'}</td>
                <td>{stats.get('max', 'N/A'):.2f if stats.get('max') is not None else 'N/A'}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_metrics_table(self, client_results: Dict) -> str:
        """Generate metrics table HTML."""
        if not client_results:
            return "<p>No client results available.</p>"
        
        html = """
        <table>
            <tr>
                <th>Client ID</th>
                <th>MAE (kW)</th>
                <th>RMSE (kW)</th>
                <th>MAPE (%)</th>
                <th>Best Trial Value</th>
            </tr>
        """
        
        for client_id, result in sorted(client_results.items()):
            metrics = result.get('metrics', {})
            html += f"""
            <tr>
                <td>{client_id}</td>
                <td>{metrics.get('mae', 'N/A'):.2f if metrics.get('mae') is not None else 'N/A'}</td>
                <td>{metrics.get('rmse', 'N/A'):.2f if metrics.get('rmse') is not None else 'N/A'}</td>
                <td>{metrics.get('mape', 'N/A'):.2f if metrics.get('mape') is not None else 'N/A'}</td>
                <td>{result.get('best_trial_value', 'N/A'):.4f if result.get('best_trial_value') is not None else 'N/A'}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_errors_table(self, errors: List) -> str:
        """Generate errors table HTML."""
        if not errors:
            return "<p>No errors encountered.</p>"
        
        html = """
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Error Type</th>
                <th>Message</th>
                <th>Recovered</th>
                <th>Recovery Action</th>
            </tr>
        """
        
        for error in errors:
            recovered_class = "error-recovered" if error.get('recovered') else "error-not-recovered"
            recovered_text = "Yes" if error.get('recovered') else "No"
            
            html += f"""
            <tr>
                <td>{error.get('timestamp', 'N/A')[:19]}</td>
                <td>{error.get('error_type', 'N/A')}</td>
                <td>{error.get('error_message', 'N/A')[:100]}</td>
                <td class="{recovered_class}">{recovered_text}</td>
                <td>{error.get('recovery_action', 'N/A')}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_trials_summary(self, trials: List) -> str:
        """Generate trials summary HTML."""
        if not trials:
            return "<p>No trials data available.</p>"
        
        # Group by client
        client_trial_counts = {}
        for trial in trials:
            client_id = trial.get('client_id', 'unknown')
            client_trial_counts[client_id] = client_trial_counts.get(client_id, 0) + 1
        
        html = f"<p>Total trials: <strong>{len(trials)}</strong></p>"
        html += f"<p>Clients with trials: <strong>{len(client_trial_counts)}</strong></p>"
        html += "<p>Average trials per client: <strong>{:.1f}</strong></p>".format(
            len(trials) / len(client_trial_counts) if client_trial_counts else 0
        )
        
        return html
    
    def _format_execution_log(self, execution_log: List) -> str:
        """Format execution log for display."""
        if not execution_log:
            return "No execution log available."
        
        lines = []
        for entry in execution_log[-100:]:  # Last 100 entries
            timestamp = entry.get('timestamp', '')[:19]
            step = entry.get('step', '')
            message = entry.get('message', '')
            lines.append(f"[{timestamp}] {step}: {message}")
        
        return '\n'.join(lines)

