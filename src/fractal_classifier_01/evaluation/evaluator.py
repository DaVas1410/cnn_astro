"""
Evaluation module for Fractal Classifier.

This module provides evaluation utilities including prediction on datasets,
centroid calculation, confusion matrix generation, and results export.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import confusion_matrix, classification_report

from .metrics import (
    compute_centroid_k,
    compute_centroid_error,
    compute_all_metrics
)

logger = logging.getLogger(__name__)


class FractalEvaluator:
    """
    Evaluator for fractal classification models.
    
    Provides methods for:
    - Batch prediction with centroid calculation
    - Evaluation on intermediate k-values
    - Confusion matrix generation
    - Results export to CSV
    """
    
    def __init__(self, 
                 model,
                 k_values: np.ndarray,
                 batch_size: int = 8):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            k_values: Array of k-values corresponding to model classes
            batch_size: Batch size for prediction
        """
        self.model = model
        self.k_values = np.array(k_values, dtype=float)
        self.batch_size = batch_size
        
        logger.info(f"FractalEvaluator initialized with k_values={k_values.tolist()}")
    
    def predict_with_centroid(self, 
                             images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict probabilities and compute centroids for images.
        
        Args:
            images: Array of images with shape (N, H, W, C)
        
        Returns:
            Tuple of:
            - probabilities: Class probabilities, shape (N, num_classes)
            - predictions: Predicted class indices, shape (N,)
            - centroids: Centroid k-values, shape (N,)
        """
        # Predict probabilities
        probabilities = self.model.predict(images, batch_size=self.batch_size, verbose=0)
        
        # Get top-1 predictions
        predictions = probabilities.argmax(axis=1)
        
        # Compute centroids
        centroids = compute_centroid_k(probabilities, self.k_values)
        
        return probabilities, predictions, centroids
    
    def evaluate_on_hdf5(self,
                        h5_path: str,
                        groups: Optional[List[str]] = None,
                        max_samples_per_group: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluate model on HDF5 dataset and compute per-image metrics.
        
        Args:
            h5_path: Path to HDF5 dataset file
            groups: List of group names to evaluate (default: all k_* groups)
            max_samples_per_group: Maximum samples per group (default: all)
        
        Returns:
            DataFrame with per-image results
        """
        logger.info(f"Evaluating on {h5_path}")
        
        rows = []
        
        with h5py.File(h5_path, 'r') as f:
            # Get groups to evaluate
            if groups is None:
                groups = sorted([g for g in f.keys() if g.startswith('k_')],
                              key=lambda s: int(s.split('_')[1]))
            
            for group in groups:
                if group not in f:
                    logger.warning(f"Group {group} not found in {h5_path}")
                    continue
                
                images_dataset = f[f'{group}/images']
                n_images = images_dataset.shape[0]
                true_k = int(group.split('_')[1])
                
                # Limit samples if requested
                if max_samples_per_group is not None:
                    n_images = min(n_images, max_samples_per_group)
                
                logger.info(f"Processing {group}: {n_images} images")
                
                # Process in batches
                for i in range(0, n_images, self.batch_size):
                    end_idx = min(i + self.batch_size, n_images)
                    batch = images_dataset[i:end_idx]
                    
                    # Ensure channel dimension
                    if batch.ndim == 3:
                        batch = batch[..., np.newaxis]
                    
                    # Predict
                    probs, preds, cents = self.predict_with_centroid(batch)
                    
                    # Get top-1 k-values and probabilities
                    top1_k = self.k_values[preds].astype(int)
                    top1_prob = probs.max(axis=1)
                    
                    # Store results for each image in batch
                    for j in range(len(cents)):
                        rows.append({
                            'group': group,
                            'image_idx': i + j,
                            'true_k': true_k,
                            'centroid_k': float(cents[j]),
                            'top1_class_idx': int(preds[j]),
                            'top1_k': int(top1_k[j]),
                            'top1_prob': float(top1_prob[j]),
                            'centroid_error': float(abs(cents[j] - true_k))
                        })
        
        df = pd.DataFrame(rows)
        logger.info(f"Evaluated {len(df)} images across {len(groups)} groups")
        
        return df
    
    def evaluate_intermediate_k(self,
                               h5_path: str,
                               output_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Evaluate model on intermediate k-values (generalization test).
        
        Args:
            h5_path: Path to HDF5 file with intermediate k-values
            output_dir: Directory to save CSV outputs (optional)
        
        Returns:
            Dictionary with 'per_image', 'readable', and 'summary' dataframes
        """
        logger.info(f"Evaluating intermediate k-values from {h5_path}")
        
        # Evaluate on all images
        df_per_image = self.evaluate_on_hdf5(h5_path)
        
        # Create readable format with additional metrics
        df_readable = self._create_readable_format(df_per_image)
        
        # Create summary statistics grouped by true_k
        df_summary = self._create_summary_statistics(df_per_image)
        
        # Save to CSV if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            df_per_image.to_csv(output_dir / 'per_image_centroids.csv', index=False)
            df_readable.to_csv(output_dir / 'intermediate_eval_readable.csv', index=False)
            df_summary.to_csv(output_dir / 'intermediate_eval_group_summary.csv')
            
            logger.info(f"Results saved to {output_dir}")
        
        return {
            'per_image': df_per_image,
            'readable': df_readable,
            'summary': df_summary
        }
    
    def _create_readable_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create readable format with error metrics."""
        df_readable = df.copy()
        
        # Add relative error
        df_readable['relative_error'] = (
            df_readable['centroid_error'] / df_readable['true_k'] * 100
        )
        
        # Add classification correctness
        df_readable['top1_correct'] = (
            df_readable['top1_k'] == df_readable['true_k']
        ).astype(int)
        
        return df_readable
    
    def _create_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics grouped by true_k."""
        summary = df.groupby('true_k').agg({
            'centroid_k': ['mean', 'std', 'min', 'max'],
            'centroid_error': ['mean', 'std'],
            'top1_prob': ['mean', 'std'],
            'image_idx': 'count'  # Number of samples
        }).round(4)
        
        # Rename count column
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        summary.rename(columns={'image_idx_count': 'n_samples'}, inplace=True)
        
        return summary
    
    def compute_confusion_matrix(self,
                                h5_path: str,
                                samples_per_class: int = 20,
                                random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute confusion matrix on training dataset samples.
        
        Args:
            h5_path: Path to HDF5 dataset (typically training set)
            samples_per_class: Number of samples per class
            random_seed: Random seed for sampling
        
        Returns:
            Tuple of:
            - confusion_matrix: Confusion matrix array
            - true_labels: True class indices
            - predictions: Predicted class indices
        """
        logger.info(f"Computing confusion matrix with {samples_per_class} samples per class")
        
        rng = np.random.RandomState(random_seed)
        X_list = []
        y_true_k = []
        
        with h5py.File(h5_path, 'r') as f:
            groups = sorted([g for g in f.keys() if g.startswith('k_')],
                          key=lambda s: int(s.split('_')[1]))
            
            for group in groups:
                images_dataset = f[f'{group}/images']
                n_images = images_dataset.shape[0]
                true_k = int(group.split('_')[1])
                
                # Sample indices
                if samples_per_class >= n_images:
                    indices = list(range(n_images))
                else:
                    indices = list(rng.choice(n_images, size=samples_per_class, replace=False))
                
                # Load images
                for idx in indices:
                    img = np.array(images_dataset[idx], dtype=np.float32)
                    if img.ndim == 2:
                        img = img[..., np.newaxis]
                    X_list.append(img)
                    y_true_k.append(true_k)
        
        X = np.stack(X_list, axis=0)
        logger.info(f"Loaded {len(X)} samples, shape: {X.shape}")
        
        # Map k-values to class indices
        k_to_idx = {int(k): i for i, k in enumerate(self.k_values)}
        y_true_idx = np.array([k_to_idx[k] for k in y_true_k], dtype=int)
        
        # Predict
        probabilities = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        y_pred_idx = probabilities.argmax(axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        
        logger.info(f"Confusion matrix computed: {cm.shape}")
        logger.info(f"Overall accuracy: {(cm.diagonal().sum() / cm.sum()):.4f}")
        
        return cm, y_true_idx, y_pred_idx
    
    def export_results(self,
                      df: pd.DataFrame,
                      output_path: str,
                      format: str = 'csv') -> None:
        """
        Export evaluation results to file.
        
        Args:
            df: DataFrame with results
            output_path: Output file path
            format: Export format ('csv' or 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Results exported to {output_path}")
    
    def print_evaluation_summary(self, df: pd.DataFrame) -> None:
        """Print summary of evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"\nTotal images evaluated: {len(df)}")
        print(f"Number of groups: {df['group'].nunique()}")
        
        # Centroid statistics
        print("\nCentroid Predictions:")
        print(f"  Mean centroid_k: {df['centroid_k'].mean():.4f} ± {df['centroid_k'].std():.4f}")
        print(f"  Min centroid_k: {df['centroid_k'].min():.4f}")
        print(f"  Max centroid_k: {df['centroid_k'].max():.4f}")
        
        # Error statistics
        print("\nCentroid Errors:")
        print(f"  Mean absolute error: {df['centroid_error'].mean():.4f}")
        print(f"  Std absolute error: {df['centroid_error'].std():.4f}")
        
        # Top-1 accuracy
        top1_correct = (df['top1_k'] == df['true_k']).mean()
        print(f"\nTop-1 Accuracy: {top1_correct:.4f} ({top1_correct*100:.2f}%)")
        
        # Per-group summary
        print("\nPer-Group Centroid Summary:")
        summary = df.groupby('group')['centroid_k'].agg(['mean', 'std', 'count']).round(4)
        print(summary)
        
        print("="*80 + "\n")
