
package nlp;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.io.File;

import com.github.lambdv.nn.MLP;
import com.github.lambdv.nn.NeuronLayer;
import com.github.lambdv.nn.Neuron;
import com.github.lambdv.nn.ActivationFunction;
import com.github.lambdv.nn.LossFunction;
import com.github.lambdv.primitives.Tensors;
import com.github.lambdv.nlp.NGram;

public class NGramTest {
    @Test
    public void basicTest() {
        var corpus = List.of(
            "i like",
            "i like cats"
        );
        var n = 2;
        
        var model = new NGram(corpus, n);

        var nextWord = model.forward("i like");
        System.out.println(nextWord);
        Assertions.assertTrue(nextWord.equals("cats"));
    }
}


