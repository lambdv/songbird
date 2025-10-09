package com.github.lambdv.nn;

import java.util.List;
import java.util.stream.IntStream;
import java.util.ArrayList;
import java.util.Random;

public class Neuron {
    private static final Random INIT_RNG = new Random(42);
    //params
    public List<Double> weights;
    public double bias;
    public ActivationFunction activation;

    //data

    public List<Double> weight_deltas;
    public double bias_delta;
    public double loss;

    public List<Double> inputCache;
    public double zCache;
    public double aCache;

    public Neuron(List<Double> weights, double bias, ActivationFunction activation){
        // Copy into a mutable list so training can update weights
        this.weights = new ArrayList<>(weights);
        this.bias = bias;
        this.activation = activation;

        this.weight_deltas = new ArrayList<>(weights.size());
        for(int i = 0; i < weights.size(); i++){
            this.weight_deltas.add(0.0);
        }
        this.bias_delta = 0.0;
        this.loss = 0.0;
        this.inputCache = new ArrayList<>(weights.size());
        this.zCache = 0.0;
        this.aCache = 0.0;
    }

    public static Neuron of(int numWeights, ActivationFunction activation){
        // Deterministic initialization with values centered around zero
        final Random r = INIT_RNG;
        return new Neuron(
            IntStream.range(0, numWeights)
                .mapToObj(i -> r.nextDouble() - 0.5) // [-0.5, 0.5)
                .toList(),
            0.0, // start biases at 0 for stability on small problems
            activation
        );
    }

    public Double forward(List<Double> input) {
        if(!validInputSize(input)){
            throw new IllegalArgumentException("Input size must match weights size");
        }
        this.inputCache = input;
        var z = IntStream.range(0, input.size())
            .mapToDouble(i -> input.get(i) * weights.get(i))
            .sum() 
            + bias;
        this.zCache = z;
        var a = activation.apply(z);
        this.aCache = a;
        return a;
    }

    public List<Double> backward(double gradOutput){
        // gradOutput is dL/da for this neuron
        double dA_dZ = activation.derivative(zCache);
        double delta = gradOutput * dA_dZ; // dL/dz

        // Accumulate parameter grads
        for(int i = 0; i < weights.size(); i++){
            double gradW = delta * inputCache.get(i); // dL/dw_i = dL/dz * x_i
            weight_deltas.set(i, weight_deltas.get(i) + gradW);
        }
        bias_delta += delta; // dL/db = dL/dz

        // Compute and return gradient w.r.t inputs to this neuron
        List<Double> gradInputs = new ArrayList<>(weights.size());
        for(int i = 0; i < weights.size(); i++){
            gradInputs.add(delta * weights.get(i)); // dL/dx_i = dL/dz * w_i
        }
        return gradInputs;
    }

    public void zeroGrad(){
        for(int i = 0; i < weight_deltas.size(); i++){
            weight_deltas.set(i, 0.0);
        }
        bias_delta = 0.0;
    }

    public void step(double learningRate){
        for(int i = 0; i < weights.size(); i++){
            weights.set(i, weights.get(i) - learningRate * weight_deltas.get(i));
        }
        bias = bias - learningRate * bias_delta;
        // After step, gradients remain accumulated unless zeroed explicitly
    }

    private boolean validInputSize(List<Double> input){
        return input.size() == weights.size();
    }
}