package com.github.lambdv.utils;

import java.util.Arrays;
import java.util.List;

public class DataProcessor {
    public static List<String> tokenize(String text) {
        if (text == null || text.isEmpty()) {
            return List.of();
        }
        return Arrays.asList(text.trim().split("\\s+"));
    }
}
