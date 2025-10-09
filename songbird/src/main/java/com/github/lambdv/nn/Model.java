package com.github.lambdv.nn;

import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

import com.github.lambdv.nn.Neuron;

public interface Model {
    public List<Double> forward(List<Double> input);
    //public void backward(Map<List<Double>, List<Double>> trainingSet);
}

// /**
//  * Model composed of sequance of fixed order models
//  */
// class ModelList<T extends Model> implements Model {
//     List<T> models;
//     public ModelList(List<T> models){
//         this.models = models;
//     }
//     public List<Double> forward(List<Double> input){
//         return models.stream()
//             .reduce(
//                 input, 
//                 (acc, model) -> model.forward(acc), 
//                 (a, b) -> b
//             );
//     }
// }

// /**
//  * Model composed of named access to models 
//  */
// class ModelMap implements Model {
//     Map<String, ? extends Model> models;
//     public ModelMap(Map<String, ? extends Model> models){
//         this.models = new TreeMap<>(models);
//     }
//     public List<Double> forward(List<Double> input){
//         return forward(input, models.keySet().stream().toList());
//     }
//     public List<Double> forward(List<Double> input, List<String> order){
//         return order.stream()
//             .map(models::get)
//             .reduce(
//                 input, 
//                 (acc, model) -> model.forward(acc), 
//                 (a, b) -> b
//             );
//     }
// }

// interface Module<
//     InputShape extends Collection<Double>, 
//     OutputShape extends Collection<Double>
// > extends Model {
//     public OutputShape forward(InputShape input);
//     public void backward(Map<InputShape, OutputShape> trainingSet);
// }
