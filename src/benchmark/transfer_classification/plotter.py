import numpy as np
from torch.utils.tensorboard import SummaryWriter
from training_state_manager import TrainingStateManager

class Plotter:
    def __init__(self, num_epochs, model_name, train_metrics, val_metrics, log_dir):
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.writer = SummaryWriter(log_dir)
        self.state_manager = TrainingStateManager()
        self.last_logged_epoch = self.state_manager.get_state_value("last_logged_epoch", 0)
        print(f"Last logged epoch: {self.last_logged_epoch}")

    def plot_metrics(self, epoch_index):
        # Adjust the epoch index for plotting (start from 1)
        adjusted_epoch_index = epoch_index + self.last_logged_epoch + 1  # Add 1 here

        self.log_to_tensorboard(adjusted_epoch_index)
        self.state_manager.set_state_value("last_logged_epoch", adjusted_epoch_index)

    def log_to_tensorboard(self, epoch_index):
        metrics_to_log = ['Loss', 'Accuracy', 'F1', 'Precision', 'Recall']
        for metric in metrics_to_log:
            # Adjust the index for zero-based arrays
            data_index = epoch_index - self.last_logged_epoch - 1

            train_value = getattr(self.train_metrics, metric.lower())[data_index]
            val_value = getattr(self.val_metrics, metric.lower())[data_index]

            self.writer.add_scalars(metric, {
                'Train': train_value,
                'Validation': val_value
            }, epoch_index)

    def close(self):
        self.writer.close()
