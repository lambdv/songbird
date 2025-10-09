package com.github.lambdv.primitives;

import java.util.List;

public record XY(
    List<List<String>> inputs, // each row's input features (strings to be parsed later)
    List<List<String>> targets // each row's target values
){}

