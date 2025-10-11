package com.github.lambdv.nlp;
import java.util.List;

import com.github.lambdv.utils.DataProcessor;
import com.github.lambdv.utils.Language;

public record NGram(List<String> corpus, int n){
    /** get most likely next word in a sequence */
    public String forward(String sequence){
        var probabilities = Language.getEnglishWords().stream()
            .map(word -> probability(sequence, word))
            .toList();
        var highestProbability = probabilities.stream().mapToDouble(x->x).max().orElse(0.0);
        var mostLikelyWord = Language.getEnglishWords().get(probabilities.indexOf(highestProbability));
        return mostLikelyWord;
    }
    /** get probability that @param word is the next word in @param sequence */
    private double probability(String sequence, String word){
        var nRecentWords = DataProcessor.tokenize(sequence)
            .subList(
                Math.max(
                    0, 
                    DataProcessor.tokenize(sequence).size() - (n() - 1)
                ), 
                DataProcessor.tokenize(sequence).size()
            )
            .stream()
            .reduce(String::concat)
            .orElse("");
        return count(nRecentWords + " " + word)+1 / count(nRecentWords)+1;
    }
    /** get number of occurrences of @param needle in the @param corpus */
    private double count(String needle){
        return corpus().stream()
            .filter(haystack -> {
                var haystackTokens = DataProcessor.tokenize(haystack);
                var needleTokens = DataProcessor.tokenize(needle);

                int matchCount = 0;
                for(int i = 0; i < Math.min(haystackTokens.size(), needleTokens.size()); i++){
                    var haystackToken = haystackTokens.get(i);
                    var needleToken = needleTokens.get(matchCount);
                    if(haystackToken.equals(needleToken)){
                        matchCount++;
                    } else {
                        matchCount = 0;
                    }
                }
                return matchCount == needleTokens.size();
            })
            .count();
    }
}

