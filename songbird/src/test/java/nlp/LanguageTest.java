package nlp;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import com.github.lambdv.utils.Language;

public class LanguageTest {
    @Test
    public void testLanguage() {
        var language = Language.getEnglishWords();
        Assertions.assertNotNull(language);
        Assertions.assertTrue(!language.isEmpty());
        Assertions.assertTrue(language.contains("hello"));
        Assertions.assertTrue(language.contains("world"));
    }
}
