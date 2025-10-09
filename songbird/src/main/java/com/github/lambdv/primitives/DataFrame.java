package com.github.lambdv.primitives;

import java.util.List;
import java.util.ArrayList;
import java.util.Optional;

public 
class DataFrame{
    List<List<String>> data;
    List<String> header;
    public DataFrame(List<List<String>> data, List<String> header){
        this.data = data;
        this.header = header;
    }
    
    public List<String> getHeader(){
        return header;
    }
    
    public List<List<String>> getData(){
        return data;
    }
    public List<String> getRow(int index){
        return data.get(index);
    }
   public List<String> getColumn(String column){
        return data.stream()
            .map(row -> row.get(header.indexOf(column)))
            .toList();
   }

   public Split split(
    double[] ratios,
    String targetColumn
   ){
    if(ratios.length != 2){
        throw new IllegalArgumentException("Only train/test split supported: provide two ratios");
    }

    double trainRatio = ratios[0];
    if(trainRatio <= 0.0 || trainRatio >= 1.0){
        throw new IllegalArgumentException("Train ratio must be in (0,1)");
    }

    int targetIdx = header.indexOf(targetColumn);
    if(targetIdx < 0){
        throw new IllegalArgumentException("Target column not found: " + targetColumn);
    }

    int numRows = data.size();
    int trainRows = (int)Math.round(numRows * trainRatio);
    if(trainRows <= 0) trainRows = 1;
    if(trainRows >= numRows) trainRows = numRows - 1;

    List<List<String>> trainInputs = new ArrayList<>();
    List<List<String>> trainTargets = new ArrayList<>();
    List<List<String>> testInputs = new ArrayList<>();
    List<List<String>> testTargets = new ArrayList<>();

    for(int i = 0; i < numRows; i++){
        var row = data.get(i);
        List<String> inputRow = new ArrayList<>();
        for(int c = 0; c < row.size(); c++){
            if(c == targetIdx) continue;
            inputRow.add(row.get(c));
        }
        List<String> targetRow = List.of(row.get(targetIdx));

        if(i < trainRows){
            trainInputs.add(inputRow);
            trainTargets.add(targetRow);
        } else {
            testInputs.add(inputRow);
            testTargets.add(targetRow);
        }
    }

    return new Split(
        new XY(trainInputs, trainTargets),
        new XY(testInputs, testTargets),
        Optional.empty()
    );
    }

   
}