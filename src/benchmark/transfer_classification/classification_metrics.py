from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

class Metrics_Micro:
    def __init__(self, y_true, y_pred, average='micro'):
        self.y_true = y_true
        self.y_pred = y_pred
        self.average = average

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average=self.average)

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average=self.average)

    def f1(self):
        return f1_score(self.y_true, self.y_pred, average=self.average)

    def roc_auc(self):
        return roc_auc_score(self.y_true, self.y_pred, average=self.average)

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred) 
    
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)