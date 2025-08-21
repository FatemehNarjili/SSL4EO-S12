import os
import torch
from tqdm import tqdm
from training_state_manager import TrainingStateManager
from plotter import Plotter
from classification_metrics import ClassificationMetrics 

class ClassificationTrainer:
    def __init__(self, model, train_dl, val_dl, criterion, optimizer, scheduler, device, saved_models_dir, num_epochs, patience, log_dir, start_epoch=0):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.device = device
        self.saved_models_dir = saved_models_dir
        self.num_epochs = num_epochs
        self.patience = patience

        self.model_name = self.model.__class__.__name__

        self.state_manager = TrainingStateManager()
        self.best_f1 = self.state_manager.get_state_value("best_f1", 0.0)
        self.epochs_no_improve = 0

        self.train_metrics = ClassificationMetrics(num_epochs)
        self.val_metrics = ClassificationMetrics(num_epochs)

        self.plotter = Plotter(
            num_epochs, 
            self.model_name,
            self.train_metrics, 
            self.val_metrics, 
            log_dir
        )
        
        # Precalculate batch counts for efficiency
        self.train_batch_count = len(train_dl)
        self.val_batch_count = len(val_dl)

        self.best_f1 = 0.0

    def train_one_epoch(self, epoch_index):
        self.model.train()
        self.train_metrics.reset()

        for image, label in tqdm(self.train_dl, desc=f"Training Epoch {epoch_index + 1}"):
            image = image.float().to(self.device)
            label = label.long().to(self.device)   # <-- FIXED

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
            label = label.long().to(self.device)   # <-- FIXED

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
            self.state_manager.set_state_value("best_f1", self.best_f1)
            self.epochs_no_improve = 0
            best_model_path = os.path.join(self.saved_models_dir, f"{self.model_name}_best.pth")
            torch.save(self.model.state_dict(), best_model_path)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping triggered at epoch {epoch_index + 1}")
            return True
        return False

    def run(self):
        for epoch_index in tqdm(range(self.num_epochs), desc="Training Progress"):
            print(f"\nEpoch {epoch_index + 1}/{self.num_epochs}")

            self.train_one_epoch(epoch_index)
            self.evaluate(epoch_index)

            self.plotter.plot_metrics(epoch_index)

            if self.scheduler:
                self.scheduler.step(self.current_val_f1)

            if self.early_stopping(epoch_index):
                break

            # Save the last model
            last_model_path = os.path.join(self.saved_models_dir, f"{self.model_name}_last_epoch.pth")
            torch.save(self.model.state_dict(), last_model_path)
            print(f"Last model saved at {last_model_path}")
        print("Training complete.")
