
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
        System.out.println(nextWord);
        Assertions.assertTrue(nextWord.equals("cats"));
    }

    @Test
    public void NGramChatBot(){
        var input = "i like";
        var corpus = List.of(
            "i like",
            "i don't like dogs",
            "i dislike cats"
        );
        var n = 2;
        var model = new NGram(corpus, n);
        var nextWord = model.forward(input);
        System.out.println(nextWord);
        Assertions.assertTrue(nextWord.equals("dogs"));
    }
    
}


