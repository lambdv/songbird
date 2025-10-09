package com.github.lambdv.primitives;

import java.util.Optional;

public record Split(
    XY train,
    XY test,
    Optional<XY> validation
){}