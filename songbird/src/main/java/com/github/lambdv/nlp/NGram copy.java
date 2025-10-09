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
        List<String> sequenceTokens = tokenize(sequence);
        int contextLength = Math.max(0, n - 1);

        List<String> context = sequenceTokens.size() <= contextLength
            ? sequenceTokens
            : sequenceTokens.subList(sequenceTokens.size() - contextLength, sequenceTokens.size());

        double numerator = countContextFollowedByWord(context, word);
        double denominator = countContext(context);

        if (denominator == 0.0) {
            return 0.0;
        }
        return numerator / denominator;
    }

    List<String> tokenize(String text) {
        if (text == null || text.isEmpty()) {
            return List.of();
        }
        return Arrays.asList(text.trim().split("\\s+"));
    }

    double countContextFollowedByWord(List<String> context, String word) {
        if (context == null) {
            context = List.of();
        }

        int contextSize = context.size();
        double count = 0.0;

        for (String sentence : corpus) {
            List<String> tokens = tokenize(sentence);

            if (contextSize == 0) {
                // Unigram case: count occurrences of the word across the corpus
                for (String token : tokens) {
                    if (token.equals(word)) {
                        count += 1.0;
                    }
                }
                continue;
            }

            for (int i = 0; i <= tokens.size() - (contextSize + 1); i++) {
                boolean match = true;
                for (int j = 0; j < contextSize; j++) {
                    if (!tokens.get(i + j).equals(context.get(j))) {
                        match = false;
                        break;
                    }
                }
                if (match && tokens.get(i + contextSize).equals(word)) {
                    count += 1.0;
                }
            }
        }
        return count;
    }

    double countContext(List<String> context) {
        if (context == null) {
            context = List.of();
        }

        int contextSize = context.size();
        double count = 0.0;

        for (String sentence : corpus) {
            List<String> tokens = tokenize(sentence);

            if (contextSize == 0) {
                // Unigram case denominator: total number of tokens
                count += tokens.size();
                continue;
            }

            for (int i = 0; i <= tokens.size() - (contextSize + 1); i++) { // ensure a next word exists
                boolean match = true;
                for (int j = 0; j < contextSize; j++) {
                    if (!tokens.get(i + j).equals(context.get(j))) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    count += 1.0;
                }
            }
        }
        return count;
    }
    
}

