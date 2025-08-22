import os
import torch
from tqdm import tqdm
from classification_metrics import ClassificationMetrics

class ClassificationTester:
    def __init__(self, model, test_dl, device, saved_models_dir):
        """
        Loads the strict 'best' checkpoint saved by the trainer:
        {ModelName}_best.pth containing: state_dict, epoch, best_f1
        """
        self.model = model.to(device)
        self.test_dl = test_dl
        self.device = device
        self.metrics = ClassificationMetrics(num_epochs=1)
        self.batch_count = len(test_dl)

        self.saved_models_dir = saved_models_dir
        self.model_name = self.model.__class__.__name__
        self.best_model_path = os.path.join(self.saved_models_dir, f"{self.model_name}_best.pth")

        # Load strict best checkpoint
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(
                f"Best checkpoint not found at '{self.best_model_path}'. "
                f"Make sure training saved it (F1 must have improved at least once)."
            )

        ckpt = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
        required = {"state_dict", "epoch", "best_f1"}
        if not isinstance(ckpt, dict) or not required.issubset(ckpt.keys()):
            raise ValueError(
                f"Invalid best checkpoint at {self.best_model_path}. "
                f"Expected keys {required}, found {set(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}."
            )

        self.model.load_state_dict(ckpt["state_dict"])
        self.best_epoch = int(ckpt["epoch"])
        self.best_f1 = float(ckpt["best_f1"])
        print(
            f"Loaded BEST model from '{self.best_model_path}' "
            f"(epoch={self.best_epoch}, best_f1={self.best_f1:.4f})."
        )

        # (Optional) where to save results if you add saving later
        self.root = os.getcwd()
        self.save_path = os.path.join(self.root, "results")
        self.index = 0

    @torch.no_grad()
    def run(self):
        self.model.eval()
        self.metrics.reset()

        for image, label in tqdm(self.test_dl, desc="Testing"):
            image = image.float().to(self.device)
            label = label.long().to(self.device)

            output = self.model(image)

            self.metrics.y_true.extend(label.cpu().numpy())
            self.metrics.y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        f1, precision, recall, accuracy, loss = self.metrics.calculate(self.batch_count)
        print(
            f"Test Metrics (best@epoch {self.best_epoch}): "
            f"accuracy={accuracy:.4f}, f1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}"
        )
