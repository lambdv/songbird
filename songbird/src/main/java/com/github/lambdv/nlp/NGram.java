package com.github.lambdv.nlp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NGram {
    final static List<String> language = new ArrayList<>();
    static {
        language.add("a");
        language.add("dog");
        language.add("cat");
        language.add("cats");

    }
    List<String> corpus;
    int n;
    public NGram(List<String> corpus, int n){
        this.corpus = corpus;
        this.n = n;
    }
    public String forward(String sequence){
        List<Double> props = language.stream()
            .map(word -> probability(sequence, word))
            .toList();
        var highestProb = props.stream()
            .mapToDouble(x->x)
            .max()
            .getAsDouble();

        var mostProbNextWord = language.get(props.indexOf(highestProb));
        return mostProbNextWord;
    }

    double probability(String sequence, String word){
        List<String> seq = Arrays.asList(sequence.split(""));
        String nRecentWords = seq.subList(seq.size() - (n+1), seq.size()-1)
            .stream()
            .reduce("", (x,y) -> x + y);
        var probabilityOfSeqWithX = count(nRecentWords + " " + word);
        var probabilityOfSeq = count(nRecentWords);
        return probabilityOfSeqWithX / probabilityOfSeq;
    }

    double count(String needle){
        return corpus.stream()
            .filter(haystack -> haystack.contains(needle))
            .count();
    }
    
}

