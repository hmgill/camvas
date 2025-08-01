import tensorflow as tf
import numpy as np

class MeanAveragePrecision(tf.keras.metrics.Metric):
    """
    Mean Average Precision metric for binary segmentation.
    
    Calculates mAP by computing Average Precision (AP) at different IoU thresholds
    and averaging them. Default uses IoU thresholds from 0.5 to 0.95 with step 0.05
    (COCO-style evaluation).
    """
    
    def __init__(self, 
                 iou_thresholds=None, 
                 num_thresholds=100,
                 name='mean_average_precision', 
                 **kwargs):
        """
        Args:
            iou_thresholds: List of IoU thresholds. If None, uses COCO-style (0.5:0.05:0.95)
            num_thresholds: Number of confidence thresholds for precision-recall curve
            name: Name of the metric
        """
        super().__init__(name=name, **kwargs)
        
        if iou_thresholds is None:
            # COCO-style IoU thresholds: 0.5, 0.55, 0.6, ..., 0.95
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = np.array(iou_thresholds)
        
        self.num_thresholds = num_thresholds
        self.confidence_thresholds = np.linspace(0, 1, num_thresholds)
        
        # Variables to store accumulated values
        self.total_ap = self.add_weight(name='total_ap', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state with batch of predictions.
        
        Args:
            y_true: Ground truth binary masks [batch, height, width] or [batch, height, width, 1]
            y_pred: Predicted probabilities [batch, height, width] or [batch, height, width, 1]
            sample_weight: Optional sample weights
        """
        # Ensure inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Squeeze if channel dimension is 1
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) == 4 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)
        
        batch_size = tf.shape(y_true)[0]
        
        # Calculate mAP for each sample in batch
        batch_aps = []
        
        for i in range(batch_size):
            true_mask = y_true[i]
            pred_probs = y_pred[i]
            
            # Calculate AP across IoU thresholds
            sample_aps = []
            
            for iou_thresh in self.iou_thresholds:
                ap = self._calculate_ap_at_iou_threshold(true_mask, pred_probs, iou_thresh)
                sample_aps.append(ap)
            
            # Average across IoU thresholds
            mean_ap = tf.reduce_mean(tf.stack(sample_aps))
            batch_aps.append(mean_ap)
        
        # Update accumulated values
        batch_map = tf.reduce_mean(tf.stack(batch_aps))
        
        if sample_weight is not None:
            batch_map = tf.reduce_mean(batch_map * sample_weight)
            weight_sum = tf.reduce_sum(sample_weight)
        else:
            weight_sum = tf.cast(batch_size, tf.float32)
        
        self.total_ap.assign_add(batch_map * weight_sum)
        self.count.assign_add(weight_sum)
    
    def _calculate_ap_at_iou_threshold(self, y_true, y_pred, iou_threshold):
        """Calculate Average Precision at a specific IoU threshold."""
        
        # Calculate precision and recall at different confidence thresholds
        precisions = []
        recalls = []
        
        for conf_thresh in self.confidence_thresholds:
            pred_mask = tf.cast(y_pred >= conf_thresh, tf.float32)
            
            # Calculate IoU
            intersection = tf.reduce_sum(y_true * pred_mask)
            union = tf.reduce_sum(tf.maximum(y_true, pred_mask))
            iou = tf.where(union > 0, intersection / union, 0.0)
            
            # Determine if prediction is positive based on IoU threshold
            is_positive = iou >= iou_threshold
            
            # Calculate precision and recall
            if tf.reduce_sum(pred_mask) > 0:
                precision = tf.where(is_positive, 1.0, 0.0)
            else:
                precision = 1.0  # No predictions made
                
            # Recall: IoU > threshold means we detected the object
            recall = tf.where(is_positive, 1.0, 0.0)
            
            precisions.append(precision)
            recalls.append(recall)
        
        precisions = tf.stack(precisions)
        recalls = tf.stack(recalls)
        
        # Calculate AP using trapezoidal rule
        # Sort by recall (though for binary segmentation this is simplified)
        ap = self._calculate_ap_from_pr_curve(precisions, recalls)
        
        return ap
    
    def _calculate_ap_from_pr_curve(self, precisions, recalls):
        """Calculate AP from precision-recall values."""
        # For binary segmentation, we simplify AP calculation
        # In practice, you might want a more sophisticated PR curve calculation
        
        # Remove duplicate recall values and sort
        unique_recalls, indices = tf.unique(recalls)
        unique_precisions = tf.gather(precisions, indices)
        
        # Sort by recall
        sorted_indices = tf.argsort(unique_recalls)
        sorted_recalls = tf.gather(unique_recalls, sorted_indices)
        sorted_precisions = tf.gather(unique_precisions, sorted_indices)
        
        # Calculate AP using trapezoidal integration
        if tf.size(sorted_recalls) < 2:
            return tf.reduce_mean(sorted_precisions)
        
        # Compute differences in recall
        recall_diffs = sorted_recalls[1:] - sorted_recalls[:-1]
        
        # Average precision over intervals
        avg_precisions = (sorted_precisions[1:] + sorted_precisions[:-1]) / 2.0
        
        # Integrate
        ap = tf.reduce_sum(recall_diffs * avg_precisions)
        
        return ap
    
    def result(self):
        """Return the current metric value."""
        return tf.where(self.count > 0, self.total_ap / self.count, 0.0)
    
    def reset_state(self):
        """Reset metric state."""
        self.total_ap.assign(0.)
        self.count.assign(0.)


# Alternative simpler implementation focusing on IoU-based mAP
class SimpleMeanAveragePrecision(tf.keras.metrics.Metric):
    """
    Simplified mAP for binary segmentation based on IoU thresholds.
    """
    
    def __init__(self, 
                 iou_thresholds=None,
                 name='simple_map',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        
        if iou_thresholds is None:
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = np.array(iou_thresholds)
        
        self.total_ap = self.add_weight(name='total_ap', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update state with IoU-based evaluation."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Threshold predictions at 0.5
        y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)
        
        # Squeeze channel dimension if present
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred_binary.shape) == 4 and y_pred_binary.shape[-1] == 1:
            y_pred_binary = tf.squeeze(y_pred_binary, axis=-1)
        
        batch_size = tf.shape(y_true)[0]
        
        # Calculate IoU for each sample
        intersection = tf.reduce_sum(y_true * y_pred_binary, axis=[1, 2])
        union = tf.reduce_sum(tf.maximum(y_true, y_pred_binary), axis=[1, 2])
        iou = tf.where(union > 0, intersection / union, 0.0)
        
        # Calculate AP as average of IoU >= threshold across all thresholds
        aps = []
        for thresh in self.iou_thresholds:
            ap = tf.reduce_mean(tf.cast(iou >= thresh, tf.float32))
            aps.append(ap)
        
        batch_map = tf.reduce_mean(tf.stack(aps))
        
        if sample_weight is not None:
            batch_map = tf.reduce_mean(batch_map * sample_weight)
            weight_sum = tf.reduce_sum(sample_weight)
        else:
            weight_sum = tf.cast(batch_size, tf.float32)
        
        self.total_ap.assign_add(batch_map * weight_sum)
        self.count.assign_add(weight_sum)
    
    def result(self):
        return tf.where(self.count > 0, self.total_ap / self.count, 0.0)
    
    def reset_state(self):
        self.total_ap.assign(0.)
        self.count.assign(0.)


# Example usage
if __name__ == "__main__":
    # Create sample data
    batch_size, height, width = 4, 128, 128
    
    # Ground truth masks (binary)
    y_true = tf.random.uniform((batch_size, height, width)) > 0.7
    y_true = tf.cast(y_true, tf.float32)
    
    # Predicted probabilities
    y_pred = tf.random.uniform((batch_size, height, width))
    
    # Initialize metrics
    map_metric = MeanAveragePrecision()
    simple_map_metric = SimpleMeanAveragePrecision()
    
    # Update metrics
    map_metric.update_state(y_true, y_pred)
    simple_map_metric.update_state(y_true, y_pred)
    
    print(f"Mean Average Precision: {map_metric.result().numpy():.4f}")
    print(f"Simple mAP: {simple_map_metric.result().numpy():.4f}")
    
    # Reset for next batch
    map_metric.reset_state()
    simple_map_metric.reset_state()
