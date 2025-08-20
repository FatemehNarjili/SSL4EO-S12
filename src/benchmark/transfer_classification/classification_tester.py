import torch
from tqdm import tqdm
from classification_metrics import ClassificationMetrics
import matplotlib.pyplot as plt
import os

class ClassificationTester:
    def __init__(self, model, test_dl, device):
        """
        Automatically set argmax_output based on the model's apply_argmax attribute.
        """
        self.model = model.to(device)
        self.test_dl = test_dl
        self.device = device
        self.metrics = ClassificationMetrics(num_epochs=1)

        self.batch_count = len(test_dl)

        self.root = os.getcwd()
        self.save_path = os.path.join(self.root, "results")
        self.index = 0

    @torch.no_grad()
    def run(self):
        self.model.eval()
        self.metrics.reset()

        for image, label in tqdm(self.test_dl):
            image = image.float().to(self.device)
            label = label.float().to(self.device)

            output = self.model(image)

            self.metrics.y_true.extend(label.cpu().numpy())  # Collect true labels

            self.metrics.y_pred.extend(torch.argmax(output, dim=1).cpu().numpy()) # Collect predictions


        f1, precision, recall, accuracy, loss = self.metrics.calculate(self.batch_count)

        print(f"Test Metrics: accuracy={accuracy:.4f}, f1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}")