<!-- 


# 音声データ(toy car data) code 

- [toySoundCheck.ipynb](toySoundCheck.ipynb)

- [script version : toySoundCheck_final.py](toySoundCheck_final.py)




# 画像データ notebook code 

- [advanced version : nutCheck.ipynb](./nutCheck.ipynb)

-  [simple version : nutcheck_easy_modifying.ipynb](./nutcheck_easy_modifying.ipynb)


# 通常の py コードに変換
- notebook code → py code 
-  [上記のコードをpy codeに変換したファイル　toySoundCheck.py](toySoundCheck.py)

-  [上記のコードをpy codeに変換したファイル　nutCheck.py](./nutCheck.py)

- 変換 command (ipynb  →　py )
            
        jupyter nbconvert --to python toySoundCheck.ipynb
        jupyter nbconvert --to python nutCheck.ipynb
        uvx --from nbconvert --with jupyter jupyter-nbconvert --to python nutcheck_easy_modifying.ipynb


        uv run toySoundCheck.py 
        uv run nutCheck.py
        uv run nutcheck_easy_modifying.py

# [show_mel_spectrogram.py](./show_mel_spectrogram.py) mel 

- result file normal vs anomaly
        
        uv run show_mel_spectrogram.py soundData/ToyCar/test/anomaly_id_04_00000000.wav

        uv run show_mel_spectrogram.py soundData/ToyCar/test/normal_id_04_00000000.wav         

- fig result 


- [normal](./soundData/ToyCar/test/normal_id_04_00000000.png)

- [anomaly](./soundData/ToyCar/test/anomaly_id_04_00000000.png)


# pure sine wave [note viz](./pure_notes_mel_spectrogram.py)

- [C, G png](./mel_C4_G4.png)

        uv run python pure_notes_mel_spectrogram.py C4 G4

# new ver -->


# 異常検知演習用コード

このフォルダには、講義で使用する **音声データ（ToyCar）** と **画像データ（Hazelnut）** の異常検知コード、および内容理解のための補助コードをまとめています。

---

## 音声データ（ToyCar）

ToyADMOS の ToyCar データを利用した異常音検知のコードです。

Notebook 版：

* [toySoundCheck.ipynb](toySoundCheck.ipynb)

講義・演習で使用する Python スクリプト版：

* [toySoundCheck_final.py](toySoundCheck_final.py)

音声ファイルから特徴量を作成し、PCA、kNN、One-Class SVM を用いて正常音と異常音を比較します。

---

## 画像データ（Hazelnut）

Autoencoder を利用した画像異常検知のコードです。

Notebook 版：

* [advanced version: nutCheck.ipynb](./nutCheck.ipynb)
* [simple version: nutcheck_easy_modifying.ipynb](./nutcheck_easy_modifying.ipynb)

`nutCheck.ipynb` は、異常スコアの工夫や詳細な評価まで含めた発展版です。

`nutcheck_easy_modifying.ipynb` は、Autoencoder による異常検知の基本的な流れを確認しやすいように簡略化したものです。

講義後半では、主に次の Python スクリプトを使用します。

* `nutcheck_easy_clean_jp.py`
* `nutcheck_apply_selected.py`

`nutcheck_easy_clean_jp.py` では、正常画像のみを用いて Autoencoder を学習し、再構成誤差から異常スコアを計算します。また、学習済みモデルや判定結果、図、CSV ファイルを保存します。

`nutcheck_apply_selected.py` は、保存済みの学習済みモデルを利用して、指定した画像だけを再度判定するためのコードです。Autoencoder を再学習せずに、別の画像について再構成結果や異常スコアを確認できます。

流れとしては、

```text
nutcheck_easy_clean_jp.py
        ↓
Autoencoder を学習
        ↓
学習済みモデルを保存
        ↓
nutcheck_apply_selected.py
        ↓
指定した画像に適用
```

となります。

---

## Notebook を Python スクリプトに変換

Notebook（`.ipynb`）は、通常の Python スクリプト（`.py`）に変換して実行することもできます。

変換済みの例：

* [toySoundCheck.py](toySoundCheck.py)
* [nutCheck.py](./nutCheck.py)

通常の Jupyter 環境では、次のように変換できます。

```bash
jupyter nbconvert --to python toySoundCheck.ipynb
jupyter nbconvert --to python nutCheck.ipynb
```

`uv` を利用する場合は、例えば次のように実行できます。

```bash
uvx --from nbconvert --with jupyter \
  jupyter-nbconvert --to python nutcheck_easy_modifying.ipynb
```

変換した Python スクリプトは、

```bash
uv run toySoundCheck.py
uv run nutCheck.py
uv run nutcheck_easy_modifying.py
```

のように実行できます。

講義後半で使用するコードについては、

```bash
uv run nutcheck_easy_clean_jp.py
```

で学習・異常判定を行い、その後、

```bash
uv run nutcheck_apply_selected.py
```

で学習済みモデルを別の画像に適用できます。

---

## Mel spectrogram の確認

音声データの特徴を視覚的に確認するための補助コードです。

* [show_mel_spectrogram.py](./show_mel_spectrogram.py)

異常音について表示する場合：

```bash
uv run show_mel_spectrogram.py \
  soundData/ToyCar/test/anomaly_id_04_00000000.wav
```

正常音について表示する場合：

```bash
uv run show_mel_spectrogram.py \
  soundData/ToyCar/test/normal_id_04_00000000.wav
```

出力例：

* [normal](./soundData/ToyCar/test/normal_id_04_00000000.png)
* [anomaly](./soundData/ToyCar/test/anomaly_id_04_00000000.png)

正常音と異常音について、時間方向・周波数方向の違いを Mel spectrogram で確認できます。

---

## Pure sine wave と Mel spectrogram

音名と周波数の関係を確認するための補助コードです。

* [pure_notes_mel_spectrogram.py](./pure_notes_mel_spectrogram.py)

例えば C4 と G4 の純音を生成して表示する場合：

```bash
uv run python pure_notes_mel_spectrogram.py C4 G4
```

出力例：

* [C4, G4 の Mel spectrogram](./mel_C4_G4.png)

純音を使うことで、音名と周波数の対応、および通常の周波数と Mel 周波数の関係を比較的単純な形で確認できます。

---

## 講義で主に使用するコード

音声データでは、

```text
toySoundCheck_final.py
```

画像データでは、

```text
nutcheck_easy_clean_jp.py
nutcheck_apply_selected.py
```

を主に使用します。

Notebook 版やその他の補助コードは、内容を詳しく確認したい場合や、追加の可視化を行いたい場合に利用します。
