import tensorflow as tf
import numpy as np

class OGMFlowMetrics:
    def __init__(self, prefix='train', no_warp=False):
        self.prefix = prefix
        self.no_warp = no_warp

        # Initialize metrics
        self.observed_auc = tf.keras.metrics.Mean(name=f'{prefix}_observed_auc')
        self.occluded_auc = tf.keras.metrics.Mean(name=f'{prefix}_occluded_auc')
        self.observed_iou = tf.keras.metrics.Mean(name=f'{prefix}_observed_iou')
        self.occluded_iou = tf.keras.metrics.Mean(name=f'{prefix}_occluded_iou')
        self.flow_epe = tf.keras.metrics.Mean(name=f'{prefix}_flow_epe')
        
        # Metrics specific to warping
        if not no_warp:
            self.flow_ogm_auc = tf.keras.metrics.Mean(name=f'{prefix}_flow_ogm_auc')
            self.flow_ogm_iou = tf.keras.metrics.Mean(name=f'{prefix}_flow_ogm_iou')
    
    def reset_states(self):
        self.observed_auc.reset_states()
        self.occluded_auc.reset_states()
        self.observed_iou.reset_states()
        self.occluded_iou.reset_states()
        self.flow_epe.reset_states()
        
        if not self.no_warp:
            self.flow_ogm_auc.reset_states()
            self.flow_ogm_iou.reset_states()
    
    def update_state(self,metrics):
        self.observed_auc.update_state(metrics.vehicles_observed_auc)
        self.occluded_auc.update_state(metrics.vehicles_occluded_auc)

        self.observed_iou.update_state(metrics.vehicles_observed_iou)
        self.occluded_iou.update_state(metrics.vehicles_occluded_iou)

        self.flow_epe.update_state(metrics.vehicles_flow_epe)
        if not self.no_warp:
            self.flow_ogm_auc.update_state(metrics.vehicles_flow_warped_occupancy_auc)
            self.flow_ogm_iou.update_state(metrics.vehicles_flow_warped_occupancy_iou)
    
    def get_results(self):
        results = {
            f'{self.prefix}_observed_auc': self.observed_auc.result().numpy(),
            f'{self.prefix}_occluded_auc': self.occluded_auc.result().numpy(),
            f'{self.prefix}_observed_iou': self.observed_iou.result().numpy(),
            f'{self.prefix}_occluded_iou': self.occluded_iou.result().numpy(),
            f'{self.prefix}_flow_epe': self.flow_epe.result().numpy()
        }
        
        if not self.no_warp:
            results.update({
                f'{self.prefix}_flow_ogm_auc': self.flow_ogm_auc.result().numpy(),
                f'{self.prefix}_flow_ogm_iou': self.flow_ogm_iou.result().numpy()
            })
        
        return results

def print_metrics(results, prefix='train', no_warp=False):
    print(f"\nMetrics for {prefix}:")
    print(f"Observed AUC: {results[f'{prefix}_observed_auc']:.4f}")
    print(f"Occluded AUC: {results[f'{prefix}_occluded_auc']:.4f}")
    print(f"Observed IOU: {results[f'{prefix}_observed_iou']:.4f}")
    print(f"Occluded IOU: {results[f'{prefix}_occluded_iou']:.4f}")
    print(f"Flow EPE: {results[f'{prefix}_flow_epe']:.4f}")
    
    if not no_warp:
        print(f"Flow OGM AUC: {results[f'{prefix}_flow_ogm_auc']:.4f}")
        print(f"Flow OGM IOU: {results[f'{prefix}_flow_ogm_iou']:.4f}")
