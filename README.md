
# 音声データ notebook code 


[toySoundCheck.ipynb](toySoundCheck.ipynb)

        
    


# 画像データ notebook code 

[nutCheck.ipynb](./nutCheck.ipynb)


# 通常の py コードに変換
- notebook code → py code 
-  [上記のコードをpy codeに変換したファイル　toySoundCheck.py](toySoundCheck.py)

-  [上記のコードをpy codeに変換したファイル　nutCheck.py](./nutCheck.py)

- 変換 command (ipynb  →　py )
            
        jupyter nbconvert --to python toySoundCheck.ipynb
        jupyter nbconvert --to python nutCheck.ipynb


        uv run toySoundCheck.py 
        uv run nutCheck.py

# [show_mel_spectrogram.py](./show_mel_spectrogram.py) mel 

- result file normal vs anomaly
        uv run show_mel_spectrogram.py soundData/ToyCar/test/anomaly_id_04_00000000.wav

        uv run show_mel_spectrogram.py soundData/ToyCar/test/normal_id_04_00000000.wav         

-  fig result 
        ./soundData/ToyCar/test/normal_id_04_00000000.png
        ./soundData/ToyCar/test/anomaly_id_04_00000000.png
