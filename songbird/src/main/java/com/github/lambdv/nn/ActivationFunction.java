package com.github.lambdv.nn;

import java.util.function.Function;
import java.util.List;
import java.util.stream.IntStream;

public interface ActivationFunction extends Function<Double, Double>{
    final static ActivationFunction ReLU = new ReLU();
    final static ActivationFunction Sigmoid = new Sigmoid();
    final static ActivationFunction Identity = new Identity();
    final static SoftMax SoftMax = new SoftMax();

    // Derivative with respect to pre-activation z
    public default double derivative(double z){
        throw new UnsupportedOperationException("Derivative not implemented for " + this.getClass().getName());
    }
}
class ReLU implements ActivationFunction{
    @Override
    public Double apply(Double input) {
        return Math.max(0, input);
    }

    @Override
    public double derivative(double z){
        return z > 0.0 ? 1.0 : 0.0;
    }
}

class Sigmoid implements ActivationFunction{
    @Override
    public Double apply(Double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derivative(double z){
        double s = apply(z);
        return s * (1.0 - s);
    }
}

class Identity implements ActivationFunction{
    @Override
    public Double apply(Double input) {
        return input;
    }

    @Override
    public double derivative(double z){
        return 1.0;
    }
}

class SoftMax implements Function<List<Double>, List<Double>>{
    
    public List<Double> apply(List<Double> z) {
        double max = z.stream().max(Double::compareTo).orElse(0.0); // for numerical stability
        double sum = z.stream().map(x -> Math.exp(x - max)).reduce(0.0, Double::sum);
        return z.stream().map(x -> Math.exp(x - max) / sum).toList();
    }

    // derivative returns the Jacobian
    public List<List<Double>> derivative(List<Double> z) {
        List<Double> s = apply(z);
        int n = z.size();
        return IntStream.range(0, n)
            .mapToObj(i -> IntStream.range(0, n)
                .mapToObj(j -> (i == j ? 1.0 : 0.0) * s.get(i) - s.get(i) * s.get(j))
                .toList())
            .toList();
    }
}