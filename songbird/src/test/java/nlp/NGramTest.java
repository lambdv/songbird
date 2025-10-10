
package nlp;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

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
        Assertions.assertTrue(nextWord.equals("cats"));
    }

    @Test
    public void NGramChatBot(){
        var input = "i like";
        var corpus = List.of(
            "i like",
            "i like cats"
        );
        var n = 2;
        var model = new NGram(corpus, n);
        var nextWord = model.forward("i like");
        Assertions.assertTrue(nextWord.equals("cats"));
    }
    
}


