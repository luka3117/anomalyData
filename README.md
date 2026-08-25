
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

