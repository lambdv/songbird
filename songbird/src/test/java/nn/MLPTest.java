package nn;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.io.File;

import com.github.lambdv.nn.MLP;
import com.github.lambdv.nn.NeuronLayer;
import com.github.lambdv.nn.Neuron;
import com.github.lambdv.nn.ActivationFunction;
import com.github.lambdv.nn.LossFunction;
import com.github.lambdv.primitives.Tensors;

public class MLPTest {
    @Test
    public void trainsXorWithGradientDescent() {
        // Architecture: 2 -> 2 -> 1 with sigmoid hidden and output
        int[] sizes = new int[]{2, 2, 1};
        ActivationFunction[] activations = new ActivationFunction[]{
            null, // unused at index 0
            ActivationFunction.Sigmoid,
            ActivationFunction.Sigmoid
        };
        MLP mlp = MLP.of(sizes, activations);

        // XOR dataset in deterministic order
        Map<List<Double>, List<Double>> data = new LinkedHashMap<>();
        data.put(List.of(0.0, 0.0), List.of(0.0));
        data.put(List.of(0.0, 1.0), List.of(1.0));
        data.put(List.of(1.0, 0.0), List.of(1.0));
        data.put(List.of(1.0, 1.0), List.of(0.0));

        // Train
        mlp.train(data, LossFunction.MeanSquaredError(), 0.3, 10000);

        // Evaluate - predictions should be close to targets
        for (var entry : data.entrySet()) {
            double pred = mlp.forward(entry.getKey()).get(0);
            double target = entry.getValue().get(0);
            // Threshold at 0.5 and also check numeric margin

            // System.err.println("pred=" + pred + ", target=" + target);
            double threshold = 0.5;
            int predictedClass = pred >= threshold ? 1 : 0;
            int targetClass = target >= threshold ? 1 : 0;
            Assertions.assertEquals(targetClass, predictedClass, "XOR class mismatch for input " + entry.getKey() + ": pred=" + pred);
            Assertions.assertTrue(Math.abs(pred - target) < threshold, "Prediction too far from target: pred=" + pred + ", target=" + target);
        }
    }

    public static double THRESHOLD = 0.5;
    @Test
    public void turseShowcase() throws Exception{
        var df = Tensors.fromFile(new File("data/xor.csv"));

        // Build model with explicit initial weights to avoid global RNG coupling across tests
        var hidden = java.util.List.of(
            new Neuron(java.util.List.of(0.5, -0.5), 0.0, ActivationFunction.Sigmoid),
            new Neuron(java.util.List.of(-0.5, 0.5), 0.0, ActivationFunction.Sigmoid)
        );
        var output = java.util.List.of(
            new Neuron(java.util.List.of(0.5, 0.5), 0.0, ActivationFunction.Sigmoid)
        );
        var model = new MLP(java.util.List.of(
            new NeuronLayer(hidden),
            new NeuronLayer(output)
        ));

        // Train on all XOR samples from CSV (no split) for determinism
        java.util.Map<java.util.List<Double>, java.util.List<Double>> trainData = new java.util.LinkedHashMap<>();
        int ix1 = df.getHeader().indexOf("x1");
        int ix2 = df.getHeader().indexOf("x2");
        int iy = df.getHeader().indexOf("output");
        for (var row : df.getData()) {
            var x = java.util.List.of(Double.parseDouble(row.get(ix1)), Double.parseDouble(row.get(ix2)));
            var y = java.util.List.of(Double.parseDouble(row.get(iy)));
            trainData.put(x, y);
        }
        model.train(trainData, LossFunction.MeanSquaredError(), 0.3, 10000);

        // Robust deterministic evaluation on all XOR inputs
        var xs = List.of(
            List.of(0.0, 0.0),
            List.of(0.0, 1.0),
            List.of(1.0, 0.0),
            List.of(1.0, 1.0)
        );
        var ys = List.of(0.0, 1.0, 1.0, 0.0);

        for (int i = 0; i < xs.size(); i++) {
            double pred = model.forward(xs.get(i)).get(0);
            double target = ys.get(i);
            int predClass = pred >= THRESHOLD ? 1 : 0;
            int targetClass = target >= THRESHOLD ? 1 : 0;
            Assertions.assertEquals(targetClass, predClass, "XOR class mismatch for input " + xs.get(i) + ": pred=" + pred);
            Assertions.assertTrue(Math.abs(pred - target) < THRESHOLD, "Prediction too far from target: pred=" + pred + ", target=" + target);
        }
    }


    @Test
    public void turseShowcase2() throws Exception{
        var df = Tensors.fromFile(new File("data/xor.csv"));

        var model = new MLP(List.of(
            new NeuronLayer(List.of(
                Neuron.of(2, ActivationFunction.Sigmoid),
                Neuron.of(2, ActivationFunction.Sigmoid)
            )),
            new NeuronLayer(List.of(
                Neuron.of(2, ActivationFunction.Sigmoid)
            ))
        ));

        // Train on all XOR samples from CSV (no split) for determinism
        java.util.Map<java.util.List<Double>, java.util.List<Double>> trainData = new java.util.LinkedHashMap<>();
        int ix1 = df.getHeader().indexOf("x1");
        int ix2 = df.getHeader().indexOf("x2");
        int iy = df.getHeader().indexOf("output");
        for (var row : df.getData()) {
            var x = java.util.List.of(Double.parseDouble(row.get(ix1)), Double.parseDouble(row.get(ix2)));
            var y = java.util.List.of(Double.parseDouble(row.get(iy)));
            trainData.put(x, y);
        }
        model.train(trainData, LossFunction.MeanSquaredError(), 0.3, 10000);

        // Robust deterministic evaluation on all XOR inputs
        var xs = List.of(
            List.of(0.0, 0.0),
            List.of(0.0, 1.0),
            List.of(1.0, 0.0),
            List.of(1.0, 1.0)
        );
        var ys = List.of(0.0, 1.0, 1.0, 0.0);

        for (int i = 0; i < xs.size(); i++) {
            double pred = model.forward(xs.get(i)).get(0);
            double target = ys.get(i);
            int predClass = pred >= THRESHOLD ? 1 : 0;
            int targetClass = target >= THRESHOLD ? 1 : 0;
            Assertions.assertEquals(targetClass, predClass, "XOR class mismatch for input " + xs.get(i) + ": pred=" + pred);
            Assertions.assertTrue(Math.abs(pred - target) < THRESHOLD, "Prediction too far from target: pred=" + pred + ", target=" + target);
        }
    }
}


