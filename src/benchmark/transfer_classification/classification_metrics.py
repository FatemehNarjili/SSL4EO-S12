import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

class ClassificationMetrics:
    def __init__(self, num_epochs, average="micro"):
        self.total_losses = 0
        self.loss = np.zeros(num_epochs)
        self.accuracy = np.zeros(num_epochs)
        self.f1 = np.zeros(num_epochs)
        self.precision = np.zeros(num_epochs)
        self.recall = np.zeros(num_epochs)
        self.average = average

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.total_losses = 0

    def update(self, epoch_index, loss, accuracy, f1, precision, recall):
        self.loss[epoch_index] = loss
        self.accuracy[epoch_index] = accuracy
        self.f1[epoch_index] = f1
        self.precision[epoch_index] = precision
        self.recall[epoch_index] = recall

    def calculate(self, batch_count):
        mean_loss =  self.total_losses / batch_count if batch_count > 0 else 0

        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, average=self.average)
        recall = recall_score(self.y_true, self.y_pred, average=self.average)
        f1 = f1_score(self.y_true, self.y_pred, average=self.average)

        return  f1, precision, recall, accuracy, mean_loss
