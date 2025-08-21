import os
from torch.utils.tensorboard import SummaryWriter

class Plotter:
    def __init__(self, num_epochs, model_name, train_metrics, val_metrics, log_dir):
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        # keep logs under model-specific subdir (optional but helpful)
        self.writer = SummaryWriter(os.path.join(log_dir, model_name))

    def plot_metrics(self, epoch_index):
        self.log_to_tensorboard(epoch_index)

    def log_to_tensorboard(self, epoch_index):
        # guard against accidental out-of-bounds
        if epoch_index < 0 or epoch_index >= self.num_epochs:
            return

        metrics_to_log = ['Loss', 'Accuracy', 'F1', 'Precision', 'Recall']
        for metric in metrics_to_log:
            key = metric.lower()  # 'Loss'->'loss', 'F1'->'f1'
            train_value = getattr(self.train_metrics, key)[epoch_index]
            val_value = getattr(self.val_metrics, key)[epoch_index]

            # Namespace by model for clarity
            tag = f"{self.model_name}/{metric}"
            self.writer.add_scalars(tag, {
                'Train': train_value,
                'Validation': val_value
            }, epoch_index)

        # optional: flush to ensure data is written promptly
        self.writer.flush()

    def close(self):
        self.writer.close()
