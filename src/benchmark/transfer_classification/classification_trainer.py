import os
import torch
from tqdm import tqdm
from plotter import Plotter
from classification_metrics import ClassificationMetrics

class ClassificationTrainer:
    def __init__(self, model, train_dl, val_dl, criterion, optimizer, scheduler,
                 device, saved_models_dir, num_epochs, patience, log_dir):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.device = device
        self.saved_models_dir = saved_models_dir
        os.makedirs(self.saved_models_dir, exist_ok=True)

        self.num_epochs = num_epochs
        self.patience = patience

        self.model_name = self.model.__class__.__name__

        # Pre-calculate batch counts
        self.train_batch_count = len(train_dl)
        self.val_batch_count = len(val_dl)

        # Paths
        self.last_model_path = os.path.join(self.saved_models_dir, f"{self.model_name}_last_epoch.pth")
        self.best_model_path = os.path.join(self.saved_models_dir, f"{self.model_name}_best.pth")

        # Training state (strict, file must store these)
        self.best_f1 = 0.0
        self.epochs_no_improve = 0
        self.start_epoch = 0  # set from checkpoint if available

        # Restore strictly from last checkpoint if it exists
        if os.path.exists(self.last_model_path):
            ckpt = torch.load(self.last_model_path, map_location=self.device, weights_only=False)
            required = {"state_dict", "epoch", "best_f1"}
            if not isinstance(ckpt, dict) or not required.issubset(ckpt.keys()):
                raise ValueError(
                    f"Invalid checkpoint at {self.last_model_path}. "
                    f"Expected keys {required}, found {set(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}."
                )
            self.model.load_state_dict(ckpt["state_dict"])
            self.start_epoch = int(ckpt["epoch"]) + 1
            self.best_f1 = float(ckpt["best_f1"])
            print(
                f"Last model loaded from '{self.last_model_path}'. "
                f"Resuming at epoch {self.start_epoch} with best_f1={self.best_f1:.4f}."
            )
        else:
            print("No previous 'last' checkpoint found. Starting from epoch 0.")

                # Metrics + Plotter
        self.train_metrics = ClassificationMetrics(self.start_epoch + self.num_epochs)
        self.val_metrics = ClassificationMetrics(self.start_epoch + self.num_epochs)
        self.plotter = Plotter(
            self.start_epoch + self.num_epochs,
            self.model_name,
            self.train_metrics,
            self.val_metrics,
            log_dir
        )


    def _save_last_checkpoint(self, epoch_index: int):
        """Save 'last' checkpoint (mandatory fields)."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "epoch": int(epoch_index),
            "best_f1": float(self.best_f1),
        }
        torch.save(checkpoint, self.last_model_path)
        print(f"Last checkpoint saved at epoch {epoch_index} -> {self.last_model_path}")

    def _save_best_checkpoint(self, epoch_index: int):
        """Save 'best' checkpoint (same metadata)."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "epoch": int(epoch_index),
            "best_f1": float(self.best_f1),
        }
        torch.save(checkpoint, self.best_model_path)
        print(f"Best model updated at epoch {epoch_index} -> {self.best_model_path}")

    def train_one_epoch(self, epoch_index):
        self.model.train()
        self.train_metrics.reset()

        for image, label in tqdm(self.train_dl, desc=f"Training Epoch {epoch_index + 1}"):
            image = image.float().to(self.device)
            label = label.long().to(self.device)

            self.train_metrics.y_true.extend(label.cpu().numpy())

            # Forward pass
            output = self.model(image)
            loss = self.criterion(output, label)

            self.train_metrics.y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            self.train_metrics.total_losses += loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

        f1, precision, recall, accuracy, loss = self.train_metrics.calculate(self.train_batch_count)
        self.train_metrics.update(epoch_index, loss, accuracy, f1, precision, recall)

        print(
            f"Train Metrics - Epoch {epoch_index + 1}: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}, f1={f1:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}"
        )

        return loss, accuracy, f1, precision, recall

    @torch.no_grad()
    def evaluate(self, epoch_index):
        self.model.eval()
        self.val_metrics.reset()

        for image, label in tqdm(self.val_dl, desc=f"Evaluating Epoch {epoch_index + 1}"):
            image = image.float().to(self.device)
            label = label.long().to(self.device)

            self.val_metrics.y_true.extend(label.cpu().numpy())

            # Forward pass
            output = self.model(image)
            loss = self.criterion(output, label)
            self.val_metrics.total_losses += loss.item()

            self.val_metrics.y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            

        f1, precision, recall, accuracy, loss = self.val_metrics.calculate(self.val_batch_count)
        self.val_metrics.update(epoch_index, loss, accuracy, f1, precision, recall)

        self.current_val_f1 = f1

        print(
            f"Val Metrics   - Epoch {epoch_index + 1}: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}, f1={f1:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}"
        )

        return loss, accuracy, f1, precision, recall

    def early_stopping(self, epoch_index):
        if self.current_val_f1 > self.best_f1:
            print(f"F1 improved from {self.best_f1:.4f} to {self.current_val_f1:.4f}. Saving model.")
            self.best_f1 = self.current_val_f1
            self.epochs_no_improve = 0
            self._save_best_checkpoint(epoch_index)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping triggered at epoch {epoch_index + 1}")
            return True
        return False

    def run(self):
        # Start at 0 if no last checkpoint; otherwise at the saved epoch index
        try:
            for epoch_index in tqdm(range(self.start_epoch, self.start_epoch + self.num_epochs), desc="Training Progress"):
                print(f"\nEpoch {epoch_index + 1}/{self.num_epochs}")

                self.train_one_epoch(epoch_index)
                self.evaluate(epoch_index)

                self.plotter.plot_metrics(epoch_index)

                if self.scheduler:
                    # Step using validation metric if your scheduler expects it
                    self.scheduler.step(self.current_val_f1)

                if self.early_stopping(epoch_index):
                    # Save a final 'last' checkpoint before exiting
                    self._save_last_checkpoint(epoch_index)
                    break

                # Save the 'last' checkpoint each epoch (weights + epoch + best_f1)
                self._save_last_checkpoint(epoch_index)
        finally:
            # Always close the plotter (flush TB) even if an error occurs
            self.plotter.close()

        print("Training complete.")
