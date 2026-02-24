class ConfusionMatrix {
  int tp = 0; // True Positive
  int tn = 0; // True Negative
  int fp = 0; // False Positive
  int fn = 0; // False Negative

  ConfusionMatrix(List<int> actual, List<int> predicted) {
    for (int i = 0; i < actual.length; i++) {
      if (actual[i] == 1 && predicted[i] == 1) tp++;
      if (actual[i] == 0 && predicted[i] == 0) tn++;
      if (actual[i] == 0 && predicted[i] == 1) fp++;
      if (actual[i] == 1 && predicted[i] == 0) fn++;
    }
  }

  double get accuracy => (tp + tn + fp + fn) == 0 ? 0 : (tp + tn) / (tp + tn + fp + fn);
  double get precision => (tp + fp) == 0 ? 0 : tp / (tp + fp);
  double get recall => (tp + fn) == 0 ? 0 : tp / (tp + fn);
  double get f1Score => (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
}
