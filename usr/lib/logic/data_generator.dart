import 'dart:math';
import 'random_forest.dart';

class DataGenerator {
  static final Random _rng = Random();

  static List<DataPoint> generateData(int count) {
    List<DataPoint> data = [];
    for (int i = 0; i < count; i++) {
      // Generate two clusters
      // Cluster 0: Centered at (0.3, 0.3)
      // Cluster 1: Centered at (0.7, 0.7)
      
      if (_rng.nextBool()) {
        // Class 0
        double x = 0.3 + _rng.nextDouble() * 0.4 - 0.2; // 0.1 to 0.5
        double y = 0.3 + _rng.nextDouble() * 0.4 - 0.2;
        data.add(DataPoint([x, y], 0));
      } else {
        // Class 1
        double x = 0.7 + _rng.nextDouble() * 0.4 - 0.2; // 0.5 to 0.9
        double y = 0.7 + _rng.nextDouble() * 0.4 - 0.2;
        data.add(DataPoint([x, y], 1));
      }
    }
    return data;
  }
}
