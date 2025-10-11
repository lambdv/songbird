package com.github.lambdv.nlp;

import java.util.Set;
import java.util.HashMap;
import java.util.function.Function;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.github.lambdv.utils.DataProcessor;

public record TermFrequency(List<String> corpus) {
    /** get term frequencies for each document */
    public List<Map<String, Integer>> frequencies(){
        var base = uniqueWords().stream().collect(Collectors.toMap(Function.identity(), word -> 0));
        return corpus.stream()
            .map(doc -> {
                Map<String, Integer> frequencies = new HashMap<>(base);
                for (String word : DataProcessor.tokenize(doc)){
                    frequencies.put(word, frequencies.get(word)+1);
                }
                return frequencies;
            })
        .toList(); 
    }

    /** get inverse document frequencies for each word */
    public Map<String, Double> inverseFrequencies(){
        return uniqueWords().stream()
            .map(word -> Map.entry(word, inverseFrequency(word)))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    /** get term weights for each document */
    public List<Map<String, Double>> termWeights(){
        return frequencies().stream()
            .map(documentTfs -> documentTfs.entrySet().stream()
                .map(tf -> Map.entry(
                    tf.getKey(),
                    tf.getValue() * inverseFrequencies().get(tf.getKey()) //tf * idf
                ))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue))
            )
            .toList();
    }
    /** get unique words in the corpus */
    public Set<String> uniqueWords(){
        return corpus.stream()
            .map(document -> DataProcessor.tokenize(document))
            .<String>mapMulti((documentWords,consumer) -> {
                documentWords.forEach(word->consumer.accept(word));
            })
            .collect(Collectors.toSet());
    }
    /** get inverse document frequency for a word */
    private double inverseFrequency(String word){
        var n = corpus().size();
        var df = corpus().stream()
            .filter(d -> DataProcessor.tokenize(d).contains(word))
            .count();
        return Math.log10((double) n / df);
    }
}
