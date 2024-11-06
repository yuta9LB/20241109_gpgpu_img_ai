# 2024/11/09 公開講座

## 使用データ
https://drive.google.com/drive/folders/1--us3n6Eo01-jsIt1RpLg5mVJhRj60vV?usp=drive_link

## 進捗
### 2024/11/05
Google Colab使い方（GoogleColabの基本的操作について.ipynb）完了<br>
CPUとGPUの比較（compare.ipynb）完了<br>
テストお願いします。<br>

### 2024/11/06
セマンティックセグメンテーション（semantic_segmentation.ipynb）<br>
テストお願いします。<br>
色々パラメータいじりながら、どれだけ精度上げられるか試してほしいです。<br>
（UNetがシンプルな構造だからか、あんまり上がらないと思いますが…）<br>
あと、余裕があれば機械学習を行う上でどんなことをしているのか、モデルがどんな構造をしているのかなど、理解するよう頑張ってみてください。<br>
一応前日準備の際、一通り説明したいと思います。<br>

<b>触れるパラメータ</b>
- 学習率: lr
- クロスエントロピー損失のクラスごとの重み: weight
- 損失のうち、Dice Lossの比率: dice_weight
- データ拡張の有無: data_augmentation
以下は時間的にも難易度的にもハードルが高いと思いますが、余裕があればぜひ試してみてください。<br>
- モデル: model
- 損失関数: criterion