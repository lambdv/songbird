package com.github.lambdv.nn;

import java.util.List;
import java.util.ArrayList;
import java.util.stream.IntStream;

public record NeuronLayer(
    List<Neuron> neurons
){
    public NeuronLayer{
        assert neurons.size() > 0 : "Neuron layer must have at least one neuron";
        assert neurons.stream().allMatch(neuron -> neuron.weights.size() == neurons.get(0).weights.size()) : "All neurons must have the same number of weights";
    }

    public static NeuronLayer of(
        int numWeights, // head/mouth
        int numNeurons, // tail
        ActivationFunction activation
    ){
        return new NeuronLayer(
            IntStream.range(0,numNeurons)
                .mapToObj(x -> Neuron.of(numWeights, activation))
                .toList()
        );
    }
    
    public List<Double> forward(List<Double> input) {
        assert input.size() == neurons.get(0).weights.size() : "Input size must match number of neurons";
        return neurons.stream()
            .map(neuron -> neuron.forward(input))
            .toList(); 
    }

    public void zeroGrad(){
        neurons.forEach(Neuron::zeroGrad);
    }

    public List<Double> backward(List<Double> gradOutput){
        assert gradOutput.size() == neurons.size() : "gradOutput size must match number of neurons";
        int numInputs = neurons.get(0).weights.size();
        List<Double> accumulatedGradInputs = new ArrayList<>(numInputs);
        for(int i = 0; i < numInputs; i++){
            accumulatedGradInputs.add(0.0);
        }
        for(int ni = 0; ni < neurons.size(); ni++){
            List<Double> gradFromNeuron = neurons.get(ni).backward(gradOutput.get(ni));
            for(int i = 0; i < numInputs; i++){
                accumulatedGradInputs.set(i, accumulatedGradInputs.get(i) + gradFromNeuron.get(i));
            }
        }
        return accumulatedGradInputs;
    }

    public void step(double learningRate){
        neurons.forEach(n -> n.step(learningRate));
    }
}