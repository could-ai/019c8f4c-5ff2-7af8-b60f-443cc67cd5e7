import 'dart:math';

class DataPoint {
  final List<double> features;
  final int label;

  DataPoint(this.features, this.label);
}

class RandomForest {
  final int numberOfTrees;
  final int maxDepth;
  final int minSamplesSplit;
  List<DecisionTree> trees = [];
  final Random _rng = Random();

  RandomForest({
    this.numberOfTrees = 10,
    this.maxDepth = 5,
    this.minSamplesSplit = 2,
  });

  void train(List<DataPoint> data) {
    trees.clear();
    if (data.isEmpty) return;
    
    for (int i = 0; i < numberOfTrees; i++) {
      // Bootstrap sample
      List<DataPoint> sample = _bootstrapSample(data);
      var tree = DecisionTree(maxDepth: maxDepth, minSamplesSplit: minSamplesSplit);
      tree.train(sample);
      trees.add(tree);
    }
  }

  int predict(List<double> features) {
    if (trees.isEmpty) return 0;
    
    int votesFor1 = 0;
    for (var tree in trees) {
      if (tree.predict(features) == 1) {
        votesFor1++;
      }
    }
    return votesFor1 >= (trees.length / 2) ? 1 : 0;
  }

  List<DataPoint> _bootstrapSample(List<DataPoint> data) {
    List<DataPoint> sample = [];
    for (int i = 0; i < data.length; i++) {
      sample.add(data[_rng.nextInt(data.length)]);
    }
    return sample;
  }
}

class DecisionTree {
  final int maxDepth;
  final int minSamplesSplit;
  Node? root;

  DecisionTree({this.maxDepth = 5, this.minSamplesSplit = 2});

  void train(List<DataPoint> data) {
    root = _buildTree(data, 0);
  }

  int predict(List<double> features) {
    return _traverse(root, features);
  }

  int _traverse(Node? node, List<double> features) {
    if (node == null) return 0;
    if (node.isLeaf) return node.label!;
    
    if (features[node.featureIndex!] <= node.threshold!) {
      return _traverse(node.left, features);
    } else {
      return _traverse(node.right, features);
    }
  }

  Node _buildTree(List<DataPoint> data, int depth) {
    int numSamples = data.length;
    int numFeatures = data.isNotEmpty ? data[0].features.length : 0;
    int numLabels1 = data.where((d) => d.label == 1).length;
    int numLabels0 = numSamples - numLabels1;
    
    // Stopping criteria
    if (depth >= maxDepth || numSamples < minSamplesSplit || numLabels1 == 0 || numLabels0 == 0) {
      return Node(isLeaf: true, label: numLabels1 > numLabels0 ? 1 : 0);
    }

    // Find best split
    double bestGini = double.infinity;
    int bestFeatureIndex = -1;
    double bestThreshold = 0.0;
    
    // For each feature
    for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
      // Get unique values for thresholds to reduce checks
      List<double> values = data.map((d) => d.features[featureIdx]).toSet().toList();
      values.sort();
      
      // Try splits between values
      for (int i = 0; i < values.length - 1; i++) {
        double threshold = (values[i] + values[i+1]) / 2;
        
        // Fast split count
        int leftCount = 0;
        int left1 = 0;
        int rightCount = 0;
        int right1 = 0;

        for (var d in data) {
          if (d.features[featureIdx] <= threshold) {
            leftCount++;
            if (d.label == 1) left1++;
          } else {
            rightCount++;
            if (d.label == 1) right1++;
          }
        }
        
        if (leftCount == 0 || rightCount == 0) continue;
        
        double gini = _calculateGiniFromCounts(leftCount, left1, rightCount, right1);
        if (gini < bestGini) {
          bestGini = gini;
          bestFeatureIndex = featureIdx;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeatureIndex == -1) {
      return Node(isLeaf: true, label: numLabels1 > numLabels0 ? 1 : 0);
    }

    // Perform the best split
    List<DataPoint> leftSplit = [];
    List<DataPoint> rightSplit = [];
    for (var d in data) {
      if (d.features[bestFeatureIndex] <= bestThreshold) {
        leftSplit.add(d);
      } else {
        rightSplit.add(d);
      }
    }

    return Node(
      isLeaf: false,
      featureIndex: bestFeatureIndex,
      threshold: bestThreshold,
      left: _buildTree(leftSplit, depth + 1),
      right: _buildTree(rightSplit, depth + 1),
    );
  }

  double _calculateGiniFromCounts(int leftSize, int left1, int rightSize, int right1) {
    double giniLeft = _gini(leftSize, left1);
    double giniRight = _gini(rightSize, right1);
    return (giniLeft * leftSize + giniRight * rightSize) / (leftSize + rightSize);
  }

  double _gini(int size, int count1) {
    if (size == 0) return 0;
    double p1 = count1 / size;
    double p0 = 1 - p1;
    return 1 - (p1 * p1 + p0 * p0);
  }
}

class Node {
  bool isLeaf;
  int? label;
  int? featureIndex;
  double? threshold;
  Node? left;
  Node? right;

  Node({required this.isLeaf, this.label, this.featureIndex, this.threshold, this.left, this.right});
}
