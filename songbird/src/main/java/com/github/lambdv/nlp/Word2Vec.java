package com.github.lambdv.nlp;
import com.github.lambdv.nn.*;
import com.github.lambdv.utils.*;

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.HashSet;

import java.util.Map;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.Arrays;

public interface Word2Vec{



/**
 * Predict center word from context word bag
 */
class CBOW implements Word2Vec {
    MLP network;
    int window = 2;
    int depth = 1;
    List<String> corpus;
    OneHotEncoder encoder;

    public CBOW(List<String> corpus){
        this.encoder = new OneHotEncoder(corpus);
        this.network = MLP.of(encoder.vocabSize(), 5, encoder.vocabSize());
        this.corpus = corpus;
    }
    
    /**
     * process corpus into dataset (input: context word, output: center word)
     */
    public static Map<List<Double>, List<Double>> processCorpus(
        OneHotEncoder encoder, List<String> corpus, int window
        ){
        var dataset = new HashMap<List<Double>, List<Double>>(); //context bag encoding to context word one hot encoding
        
        for (String document : corpus){
            var words = DataProcessor.tokenize(document);
            
            for(int i = 0; i < words.size(); i++){
                var contextWords = words.subList(Math.max(0, i - window), Math.min(words.size(), i + window + 1));
                var contextBag = encoder.oneHotBag(contextWords);
                var centerWordOneHot = Arrays.stream(encoder.oneHot(words.get(i))).mapToObj(Double::valueOf).toList();
                dataset.put(Arrays.stream(contextBag).mapToObj(Double::valueOf).toList(), centerWordOneHot);
            }
        }
        return dataset;
    }

    public void train(List<String> corpus){
        var dataset = processCorpus(encoder, corpus, window);
        network.train(dataset, LossFunction.MeanSquaredError(), 0.01, 1000);
    }

    /**
     * predict center word from context words
     * @param contextWordBag
     * @return
     */
    public String forward(List<String> contextWordBag){
        var x = Arrays.stream(encoder.oneHotBag(contextWordBag)).mapToObj(Double::valueOf).toList();
        var y = network.forward(x);
        var highestProbability = y.stream().max(Double::compareTo).orElse(0.0);
        var index = y.indexOf(highestProbability);
        if (index == -1) {
            throw new RuntimeException("No one-hot vector found for index " + index);
        }
        return encoder.decode(index);
    }
}

class OneHotEncoder {
    private final Map<String, Integer> encoding; // word -> index
    private final Map<Integer, String> decoding; // index -> word

    public OneHotEncoder(List<String> corpus) {
        var i = new AtomicInteger();
        // Extract all unique tokens from the corpus
        Set<String> vocab = corpus.stream()
                .map(DataProcessor::tokenize)
                .flatMap(List::stream)
                .collect(Collectors.toSet());

        // Build encoding (word -> index)
        encoding = vocab.stream()
                .collect(Collectors.toMap(
                        word -> word,
                        word -> i.getAndIncrement()
                ));

        // Build decoding (index -> word)
        decoding = encoding.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
    }

    /**
     * Returns the index representing this word in the vocabulary.
     * You can use this index to create a one-hot vector.
     */
    public Integer encode(String word) {
        return encoding.get(word);
    }

    /**
     * Returns the word corresponding to a given index.
     */
    public String decode(int index) {
        return decoding.get(index);
    }

    public String decodeOneHot(double[] oneHot) {
        var oneHotList = Arrays.stream(oneHot).mapToObj(Double::valueOf).toList();
        //assert oneHotList.size() == encoding.size();
        var index = oneHotList.indexOf(1.0);
        if (index == -1) {
            throw new RuntimeException("No one-hot vector found for index " + index);
        }
        return decode(index);
    }

    /**
     * Returns the one-hot vector representation as a double array.
     * Useful for Skip-Gram input/output encoding.
     */
    public double[] oneHot(String word) {
        double[] vector = new double[encoding.size()];
        Integer idx = encode(word);
        if (idx != null)
            vector[idx] = 1.0;
        return vector;
    }

    public double[] oneHotBag(List<String> words) {
        BiFunction<double[], double[], double[]> or = (a, b) -> {
            double[] result = new double[a.length];
            for(int i = 0; i < a.length; i++){
                if (a[i] == 1.0 | b[i] == 1.0){
                    result[i] = 1.0;
                } else {
                    result[i] = 0.0;
                }
            }
            return result;
        };
        return words.stream()
            .map(this::oneHot)
            .reduce((x,y) -> or.apply(x,y))
            .orElseThrow(() -> new RuntimeException("No words provided"));
    }

    /**
     * Returns vocabulary size.
     */
    public int vocabSize() {
        return encoding.size();
    }
}

}