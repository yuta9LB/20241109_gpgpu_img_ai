# 2024/11/09 公開講座

## ファイル一覧
### Jupiter Notebook
↓ Google Colab
- 0_GoogleColabの基本的操作について.ipynb
    - 1_compare.ipynbに統合予定
- 1_compare.ipynb
    - CPUとGPUの比較
    - Google Colab側で調整必要
- 2_machine_learning.ipynb
    - 機械学習・セマンティックセグメンテーションについての説明
    - Google Colab側で調整必要
    - アンケート結果によっては調整必要
↓ 以降クラスター側
- 3-1_semantic_segmentation_with_unet.ipynb
    - UNetの訓練
    - 可視化の部分調整予定
- 3-2_semantic_segmentation_with_pspnet.ipynb
    - PSPNetの訓練
    - 可視化の部分調整予定
- 4_test.ipynb
    - 各グループでの推論可視化＋最終評価用
    - 評価計算法変更予定
- 5_hints.ipynb
    - 演習のヒント
    - 未了

### Pythonスクリプト
- unet.py
    - UNetモデル
    - 演習の際いじってもらってOK
- pspnet.py
    - PSPNetモデル
    - 演習の際いじってもらってOK
- train_u.py
    - UNetの訓練スクリプト
    - JupiterじゃなくてこっちでもOK
- train_psp.py
    - PSPNetの訓練スクリプト
    - JupiterじゃなくてこっちでもOK
- dataset.py
    - データセットスクリプト
    - 演習時前処理をいじってもいいかも
- util.py
    - その他関数等
    - 現状ロスだけなので、'loss.py'とかにするかも

### その他
- requirements.txt
    - Colabでもクラスターでも最初に'pip install -r requirements.txt'してね
- img/
    - Jupiterノートに載せている画像を入れているディレクトリ
- weights/
    - 重みの初期値を入れておくディレクトリ
    - 重いのでgit経由無理、手作業で用意
- VOCdevkit/
    - データセットディレクトリ
