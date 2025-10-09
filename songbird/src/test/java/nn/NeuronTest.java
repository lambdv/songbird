package nn;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import com.github.lambdv.nn.Model;
import com.github.lambdv.nn.Neuron;
import com.github.lambdv.nn.ActivationFunction;
import java.util.ArrayList;
import com.github.lambdv.nn.ActivationFunction;
import java.util.List;

public class NeuronTest {
    // @Test public void creatingNeuronUnitTest() {
    //     List<Double> weights = List.of(
    //         1.0,1.0
    //     );
    //     Neuron n = new Neuron(weights, 0.0, ActivationFunction.ReLU); 
    //     Neuron n2 = Neuron.of(weights.size(), ActivationFunction.ReLU);
    //     Assertions.assertEquals(n.weights, n2.weights);
    // }

    @Test
    public void neuronForwardUnitTest() {
        var n = new Neuron(List.of(1.0,1.0), 0.0, ActivationFunction.Identity);
        var res = n.forward(List.of(1.0,1.0));
        Assertions.assertEquals(res, 2.0);

        var n2 = new Neuron(List.of(1.0,1.0), 1.0, ActivationFunction.Identity);
        var res2 = n2.forward(List.of(1.0,1.0));
        Assertions.assertEquals(res2, 3.0);

        var n3 = new Neuron(List.of(2.0,3.0), 1.0, ActivationFunction.Identity);
        var res3 = n3.forward(List.of(4.0,5.0));
        Assertions.assertEquals(res3, 2*4 + 3*5 + 1);


    }
}
