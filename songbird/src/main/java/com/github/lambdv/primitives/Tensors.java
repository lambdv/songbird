package com.github.lambdv.primitives;

import java.util.List;
import java.util.ArrayList;
import java.nio.file.Files;
import java.io.File;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import java.nio.charset.StandardCharsets;

public class Tensors {
    public static DataFrame fromFile(File file) throws Exception {
        try (var reader = Files.newBufferedReader(file.toPath(), StandardCharsets.UTF_8)) {
            var parser = CSVParser.parse(reader, CSVFormat.DEFAULT);
            var records = parser.getRecords();

            if (records.isEmpty()) {
                throw new IllegalArgumentException("CSV has no records: " + file);
            }

            var first = records.get(0);
            boolean firstRowIsNumeric = first.stream().allMatch(v -> {
                try {
                    Double.parseDouble(v);
                    return true;
                } catch (Exception e) {
                    return false;
                }
            });

            List<CSVRecord> dataRecords;
            List<String> header;

            if (firstRowIsNumeric) {
                int numCols = first.size();
                header = new ArrayList<>(numCols);
                for (int i = 0; i < numCols - 1; i++) header.add("x" + (i + 1));
                header.add("output");
                dataRecords = records;
            } else {
                header = first.toList();
                dataRecords = records.subList(1, records.size());
            }

            var data = dataRecords.stream().map(CSVRecord::toList).toList();
            return new DataFrame(data, header);
        }
    }
}


