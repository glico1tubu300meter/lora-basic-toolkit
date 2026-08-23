# lora-basic-toolkit

QLoRA(4bit量子化 + LoRA)によるローカルLLMファインチューニングの最小構成ツールキット。学習データは外部のJSONLファイルとして自分で用意する形式で、特定の作品・キャラクターなど著作権・商標に関わる内容は一切含まれていない。

対象モデルはデフォルトで [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)。`--model` オプションで他の Hugging Face モデルにも変更できる。

## 動作環境

- NVIDIA GPU (CUDA)、VRAM 11GB以上を推奨(7Bモデル・4bit量子化での動作確認は GTX 1080 Ti で実施)
- Windows / Linux で動作確認
- Python 3.11

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate   # Windows は .venv\Scripts\activate
pip install -r requirements.txt
```

## データ形式

学習データは JSONL(1行1件)、`prompt` と `response` のペア:

```json
{"prompt": "ユーザーの発言", "response": "モデルに学習させたい応答"}
```

`data/sample_data.jsonl` に、フォーマット確認用の汎用サンプル(5件、著作権・商標フリー)を同梱している。実際に使う際は、権利上問題のない自前のデータに差し替えること。

## 使い方

### 1. 学習

```bash
cd scripts
python train_lora.py --data ../data/sample_data.jsonl --output_dir ../out
```

主なオプション:

| オプション | デフォルト | 内容 |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | ベースモデル |
| `--r` | 16 | LoRAランク |
| `--alpha` | 32 | LoRAスケーリング係数 |
| `--epochs` | 3.0 | 学習エポック数 |
| `--lr` | 2e-4 | 学習率 |
| `--batch_size` | 2 | バッチサイズ |
| `--use_dora` | (フラグ) | DoRAに切り替える |

学習後、`<output_dir>/lora_adapter/` にアダプタが保存される。

### 2. 推論(Base/LoRA比較)

```bash
python infer_lora.py --adapter ../out/lora_adapter "質問文をここに"
```

## 実行イメージ

同梱の `sample_data.jsonl`(5件)を3エポック学習させた際の実際の出力例:

```
$ python train_lora.py --data ../data/sample_data.jsonl --output_dir ../out --epochs 3

trainable params: 40,370,176 || all params: 7,655,986,688 || trainable%: 0.5273
Loaded 5 training examples from ../data/sample_data.jsonl

{'loss': '3.41', 'grad_norm': '5.312', 'learning_rate': '0.0002', 'mean_token_accuracy': '0.5667', 'epoch': '1'}
{'loss': '2.205', 'grad_norm': '2.641', 'learning_rate': '0.0001333', 'mean_token_accuracy': '0.6839', 'epoch': '2'}
{'loss': '1.764', 'grad_norm': '2.609', 'learning_rate': '6.667e-05', 'mean_token_accuracy': '0.7419', 'epoch': '3'}
{'train_runtime': '36.27', 'train_samples_per_second': '0.414', 'train_loss': '2.46', 'epoch': '3'}

Saved LoRA adapter to: ../out/lora_adapter
```

続けて推論を実行すると、Base(素のモデル)と LoRA適用後で応答の傾向が変わっていることが確認できる:

```
$ python infer_lora.py --adapter ../out/lora_adapter "おすすめの本を教えて"

[base] もちろん、おすすめの本をいくつか紹介させていただきます。読むジャンルや興味がある
特定のトピックがある場合は、それを教えていただければ、よりぴったりの本を提案できるかも
しれません。

1. **ビジネス/経営**:
   - 「リーダーシップの科学」（アレックス・オステンロス）
   ...

[lora] どのようなジャンルやテーマがお好みでしょうか？それによっておすすめの本が変わる
可能性があります。例えば、ビジネス書、フィクション、科学技術、歴史など、何について
学びたいか教えていただけますと、より適した本を提案できます。
```

`sample_data.jsonl` には「まず相手の希望を聞き返してから答える」という応答パターンの例が含まれており、わずか5件・3エポックの学習でもその傾向がLoRA適用後の応答に反映されているのが分かる。

## 既知の注意点

- **Windows + mmap の相性問題**: ネットワークドライブ等にモデルキャッシュを置いた環境で、7B級モデルを `from_pretrained` でロードするとセグメンテーション違反が発生することがある。本スクリプトでは `disable_mmap=True` を指定して回避している。
- **Pascal世代GPU(GTX 10xx等)でのbf16非対応**: Qwen2.5系モデルの既定dtypeはbfloat16だが、Pascal世代GPUはbf16の勾配スケーリング(AMP + GradScaler)に対応していない。本スクリプトでは `torch.set_default_dtype(torch.float16)` と `fp16=False, bf16=False`(LoRAアダプタ自体はfp32のまま学習)を組み合わせて回避している。より新しいGPU(Ampere以降)では `fp16=True` に変更した方が高速な場合がある。

## ライセンス

このリポジトリのコード自体に付属する追加のライセンス条件はない。使用するベースモデル(Qwen2.5シリーズ等)のライセンス、および自分で用意する学習データの権利については、各自で確認すること。
