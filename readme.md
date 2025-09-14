# このリポジトリについて
LLMの学習コードをまとめた。

## 環境構築
- Python: 3.12
- その他ライブラリ
```
pip install -r requirements.txt
```
### uv による環境構築
- python のバージョン変更
```
uv python pin 3.12
```
- 仮想環境の生成
```
uv venv
```
- 仮想環境の起動(.venvの場合)
```
. .venv/bin/activate
```
- ライブラリのインストール
```
uv pip  install -r requirements.txt
```

## 実行方法
標準出力と標準エラー出力を別ファイルで書き出したいときは、それぞれのコマンドの後ろに以下を追加する
```
>1 stdout.log 2> stderr.log
```
- sft.py
```
python sft.py --config configs/sft.yaml
```
