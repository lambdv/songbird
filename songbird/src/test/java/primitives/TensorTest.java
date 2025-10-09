package primitives;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import com.github.lambdv.nn.Model;
import com.github.lambdv.nn.Neuron;
import com.github.lambdv.nn.ActivationFunction;
import java.util.ArrayList;
import com.github.lambdv.nn.ActivationFunction;
import java.util.List;
import com.github.lambdv.primitives.Tensor;
import java.util.Arrays;
import java.util.stream.IntStream;


public class TensorTest {
    @Test
    public void testTensor() {
        Tensor t = Tensor.zeros(2, 2);
        int[] expectedShape = {2, 2};
        Assertions.assertArrayEquals(expectedShape, t.shape());

        t.set(1.0, 0, 0);
        Assertions.assertEquals(1.0, t.get(0, 0));

        System.out.println(t.toStringShape());
        // var v = t.getSlice(0);
        // System.out.println(Arrays.toString(v));
    }

    @Test
    public void testTensor2() {
        Tensor t = Tensor.zeros(2, 6);
        System.out.println(t.toStringShape());

        var x = t.getSlice(0);
        System.out.println(x.toStringShape());

        var expected = IntStream.range(0,6)
            .mapToObj(i -> 0.0)
            .toArray();
        Assertions.assertEquals(Arrays.toString(expected), Arrays.toString(x.toArray()));
    }

    @Test public void testToListMethod() {
        Tensor t = Tensor.zeros(2, 6);
        List<?> l = t.toList();

        Object res = l.get(0);
        //Assertions.assertTrue(res instanceof List<?> x && x instanceof List<Double>);
    }
}
 