package com.github.lambdv.nn;

import java.util.function.BiFunction;

public interface LossFunction extends BiFunction<Double, Double, Double>{

    public Double backward(Double predicted, Double target);

    public default Double forward(Double predicted, Double target){
        return apply(predicted, target);
    }

    public static LossFunction MeanSquaredError(){
        return MSELoss;
    }
    public static LossFunction CrossEntropy(){
        return CrossEntropyLoss;
    }
    public static LossFunction BinaryCrossEntropy(){
        return BinaryCrossEntropyLoss;
    }
    public static LossFunction SoftmaxCrossEntropy(){
        return SoftmaxCrossEntropyLoss;
    }

    static LossFunction MSELoss = new MeanSquaredError();
    static LossFunction CrossEntropyLoss = new CrossEntropyLoss();
    static LossFunction BinaryCrossEntropyLoss = new BinaryCrossEntropyLoss();
    static LossFunction SoftmaxCrossEntropyLoss = new SoftmaxCrossEntropyLoss();
}

//linear regression
class MeanSquaredError implements LossFunction{
    @Override
    public Double apply(Double predicted, Double target) {
        return Math.pow(predicted - target, 2);
    }
    
    public Double backward(Double predicted, Double target){
        return 2 * (predicted - target);
    }
}

//classification/logistic regression
class CrossEntropyLoss implements LossFunction{
    @Override
    public Double apply(Double predicted, Double target) {
        return -Math.log(predicted);
    }
    public Double backward(Double predicted, Double target){
        return -1 / predicted;
    }
}

//binary classification (sigmoid)
class BinaryCrossEntropyLoss implements LossFunction{
    @Override
    public Double apply(Double predicted, Double target) {
        return -Math.log(predicted);
    }
    public Double backward(Double predicted, Double target){
        return -1 / predicted;
    }
}

//multi-class classification
class SoftmaxCrossEntropyLoss implements LossFunction{
    @Override
    public Double apply(Double predicted, Double target) {
        return -Math.log(predicted);
    }
    public Double backward(Double predicted, Double target){
        return -1 / predicted;
    }
}