package nlp;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import com.github.lambdv.utils.Language;
import com.github.lambdv.nlp.*;
import java.util.Set;
import java.util.Map;
import java.util.HashMap;

import java.util.List;
import java.util.ArrayList;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class TermFrequencyTest {
    @Test
    public void testUnqiueMethod() {
        var corpus = List.of(
            "test rubber",
            "test duck"
        );
        var tf = new TermFrequency(corpus);
        
        var res = tf.uniqueWords();

        Assertions.assertEquals(
            res, 
            Set.of("test", "duck", "rubber")
        );
    }


    @Test
    public void testFrequenciesMethod() {
        var tf = new TermFrequency(List.of(
            "this is the first document",
            "this document is the second document",
            "and this is the third one",
            "is this the first one"
        ));
        
        var res = tf.frequencies();
        var expected = List.of(
            Map.of("and", 0, "document", 1, "first", 1, "is", 1, "one", 0, "second", 0, "the", 1, "third", 0, "this", 1),
            Map.of("and", 0, "document", 2, "first", 0, "is", 1, "one", 0, "second", 1, "the", 1, "third", 0, "this", 1),
            Map.of("and", 1, "document", 0, "first", 0, "is", 1, "one", 1, "second", 0, "the", 1, "third", 1, "this", 1),
            Map.of("and", 0, "document", 0, "first", 1, "is", 1, "one", 1, "second", 0, "the", 1, "third", 0, "this", 1)
        );
        Assertions.assertEquals(res, expected);
        
    }

    @Test
    public void testTermWeightsMethod() {
        var tf = new TermFrequency(List.of(
            "this is the first document",
            "this document is the second document",
            "and this is the third one",
            "is this the first one"
        ));
        
        var res = tf.termWeights();
        var expected = List.of(
            Map.of("and", 0.0, "document", 0.301, "first", 0.301, "is", 0.0, "one", 0.0, "second", 0.0, "the", 0.0, "third", 0.0, "this", 0.0),
            Map.of("and", 0.0, "document", 0.602, "first", 0.0, "is", 0.0, "one", 0.0, "second", 0.602, "the", 0.0, "third", 0.0, "this", 0.0),
            Map.of("and", 0.602, "document", 0.0, "first", 0.0, "is", 0.0, "one", 0.301, "second", 0.0, "the", 0.0, "third", 0.602, "this", 0.0),
            Map.of("and", 0.0, "document", 0.0, "first", 0.301, "is", 0.0, "one", 0.301, "second", 0.0, "the", 0.0, "third", 0.0, "this", 0.0)
        );

        Assertions.assertEquals(res.size(), expected.size());

        for(int i = 0; i < res.size(); i++){
            var docWeights = res.get(i);
            for(var weight : docWeights.entrySet()){
                var expectedWeight = expected.get(i).get(weight.getKey());
                Assertions.assertEquals(weight.getValue(), expectedWeight, 0.1);
            }
        }
    }
}