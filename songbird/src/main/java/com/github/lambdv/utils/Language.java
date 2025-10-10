package com.github.lambdv.utils;

import java.io.File;
import java.io.FileOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import org.apache.commons.io.IOUtils;

public class Language {
    private static Optional<List<String>> language = Optional.empty();
    
    public final static List<String> getEnglishWords(){
        if (language.isEmpty()){
            var file = new File("data/english.txt");
            if(!file.exists()){
                downloadEnglishWordsDataSource();
            }
            try{
                language = Optional.of(Files.readAllLines(Path.of(file.getPath())));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        return language.get();
    }
    
    public final static void downloadEnglishWordsDataSource(){
        try{
            var url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt";
            var file = new File("data/english.txt");
            var urlConnection = (HttpURLConnection) new URL(url).openConnection();
            urlConnection.setRequestMethod("GET");
            urlConnection.connect();
            var inputStream = urlConnection.getInputStream();
            var outputStream = new FileOutputStream(file);
            IOUtils.copy(inputStream, outputStream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
