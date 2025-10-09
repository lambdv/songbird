# SongBird
a high-level, object oriented deep learning framework.

```java
var data = Tensors //load data
    .fromFile(new File("data/xor.csv"))
    .split(new double[]{0.8, 0.2}, "output");

var model = MLP.of( //build model
    new int[]{2,2,1}, 
    new ActivationFunction[]{ActivationFunction.Identity, ActivationFunction.Sigmoid, ActivationFunction.Sigmoid}
);

double learningRate = 0.3;
double epoch = 10000;
double cost = LossFunction.MeanSquaredError();
model.train(data.train(), cost, learningRate, epoch); //train

double accuracy = model.test(data.test()); //test
assert accuracy > 0.9;

var x = List.of(0.0, 1.0);
var y = model.forward(x).get(0); //use
assert 1.0 == y;
```

## Set Up
```bash
git clone http://github.com/lambdv/songbird
cd songbird
mvn install
mvn test
```