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

public class Word2VecTest {
    @Test
    public void integrationTestCBOW() {
        var corpus = List.of(
            "i like drinking water",
            "i like drinking coffee",
            "i don't like drinking tea",
            "i like eating rice",
            "i like eating bread"
        );
        var w2v = new Word2Vec.CBOW(corpus);
        w2v.train(corpus);
        var res = w2v.forward(List.of("i", "drinking"));
        Assertions.assertEquals(res, "like");
    }

    @Test
    public void unitTestOneHotEncoder() {
        var encoder = new Word2Vec.OneHotEncoder(List.of("i", "like", "drinking", "water", "coffee", "tea", "rice", "bread"));
        Assertions.assertEquals(encoder.encode("i"), 0);
        Assertions.assertEquals(encoder.encode("like"), 1);
        Assertions.assertEquals(encoder.encode("drinking"), 2);
        Assertions.assertEquals(encoder.encode("water"), 3);
        Assertions.assertEquals(encoder.encode("coffee"), 4);
        Assertions.assertEquals(encoder.encode("tea"), 5);
        Assertions.assertEquals(encoder.encode("rice"), 6);
        Assertions.assertEquals(encoder.encode("bread"), 7);
    }
}