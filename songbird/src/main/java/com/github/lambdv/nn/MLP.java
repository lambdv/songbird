package com.github.lambdv.nn;

import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.ArrayList;
import com.github.lambdv.primitives.XY;

public class MLP{
    List<NeuronLayer> layers;
    public MLP(List<NeuronLayer> layers){
        this.layers = layers;
    }

    public static MLP of(int... sizes){
        return new MLP(
            IntStream.range(1, sizes.length)
                .mapToObj(i -> NeuronLayer.of(sizes[i-1], sizes[i], ActivationFunction.Identity))
                .toList()
        );
    }

    public static MLP of(int[] sizes, ActivationFunction[] activations){
        return new MLP(
            IntStream.range(1, sizes.length)
                .mapToObj(i -> NeuronLayer.of(sizes[i-1], sizes[i], activations[i]))
                .toList()
        );
    }



    /**
     * feeds input through the network and returns the output
     * @param input
     * @return
     */
    public List<Double> forward(List<Double> input){
        return layers.stream()
            .reduce(
                input, 
                (acc, layer) -> layer.forward(acc), 
                (a, b) -> b
            );
    }

    

    /**
     * propagates gradient through the network backwards
     * @param gradOutput
     * @return
     */
    public List<Double> backward(List<Double> gradient){
        return layers.reversed().stream()
            .reduce(gradient, (previousGradient, layer) -> layer.backward(previousGradient), (a, b) -> b);
    }

    /**
     * resets the gradients of the network
     */
    public void zeroGrad(){
        layers.forEach(NeuronLayer::zeroGrad);
    }

    /**
     * updates the weights of the network
     * @param learningRate
     */
    public void step(double learningRate){
        layers.forEach(l -> l.step(learningRate));
    }
    
    public void train(Map<List<Double>, List<Double>> trainingSet, LossFunction lossFunction, double learningRate, int epochs){
        var samples = trainingSet.entrySet().stream().toList();
        if(samples.isEmpty()) return;

        for(int epoch = 0; epoch < epochs; epoch++){
            for(var sample : samples){
                var input = sample.getKey();
                var target = sample.getValue();

                // SGD per-sample style: zero grads, forward, backward, step
                zeroGrad();
                var output = forward(input);

                List<Double> gradOut = new ArrayList<>(output.size());
                for(int i = 0; i < output.size(); i++){
                    gradOut.add(lossFunction.backward(output.get(i), target.get(i)));
                }

                backward(gradOut);
                step(learningRate);
            }
        }
    }

    public void train(XY trainingSet, LossFunction lossFunction, double learningRate, int epochs){
        var inputs = trainingSet.inputs();
        var targets = trainingSet.targets();
        if(inputs.size() != targets.size()) return;

        for(int epoch = 0; epoch < epochs; epoch++){
            for(int idx = 0; idx < inputs.size(); idx++){
                var input = inputs.get(idx).stream().map(Double::parseDouble).toList();
                var target = targets.get(idx).stream().map(Double::parseDouble).toList();

                zeroGrad();
                var output = forward(input);

                List<Double> gradOut = new ArrayList<>(output.size());
                for(int i = 0; i < output.size(); i++){
                    gradOut.add(lossFunction.backward(output.get(i), target.get(i)));
                }

                backward(gradOut);
                step(learningRate);
            }
        }
    }


    public double test(XY testSet){
        return test(testSet, 0.5);
    }

    public double test(XY testSet, double threshold){
        var inputs = testSet.inputs();
        var targets = testSet.targets();
        //if(inputs.isEmpty() || inputs.size() != targets.size()) return 0.0;

        int correct = 0;
        for(int idx = 0; idx < inputs.size(); idx++){
            var input = inputs.get(idx).stream().map(Double::parseDouble).toList();
            var target = targets.get(idx).stream().map(Double::parseDouble).toList();

            var output = forward(input);

            if(output.size() == 1 && target.size() == 1){
                int predClass = output.get(0) >= threshold ? 1 : 0;
                int targetClass = target.get(0) >= threshold ? 1 : 0;
                if(predClass == targetClass) correct++;
            } else {
                int predIdx = argmax(output);
                int targetIdx = argmax(target);
                if(predIdx == targetIdx) correct++;
            }
        }
        return (double) correct / inputs.size();
    }

    private static int argmax(List<Double> values){
        int bestIdx = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < values.size(); i++){
            double v = values.get(i);
            if(v > bestVal){
                bestVal = v;
                bestIdx = i;
            }
        }
        return bestIdx;
    }


} 