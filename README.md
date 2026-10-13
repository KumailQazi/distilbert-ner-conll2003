# DistilBERT token classification on CoNLL-2003

A reproducible CPU-friendly fine-tuning project for English named-entity recognition (NER). It fine-tunes `distilbert-base-uncased` on CoNLL-2003 with token-level supervision and reports entity-level precision, recall, F1, token accuracy, validation loss, and test metrics.

> **Scope boundary for the Arabic-diacritization application:** This is adjacent evidence of transformer sequence labeling only. NER is not Arabic diacritization, does not reproduce the client's character-level encoder-decoder/CATT recipe, does not measure DER/WER, and does **not** enforce the invariant that base letters cannot change.

## Measured results

Completed three epochs with seed 42, batch size 8, max sequence length 128, and learning rate `5e-5`. The best checkpoint was selected on validation F1; the official test split was evaluated afterward.

| Split | Precision | Recall | F1 | Loss |
|---|---:|---:|---:|---:|
| Validation | 0.93850 | 0.94198 | **0.94024** | 0.05734 |
| Test | 0.89458 | 0.90204 | **0.89829** | 0.14254 |

See [`results.md`](results.md) for sample counts, token accuracy, hardware and timing. These results were measured on the available Linux CPU environment.

## Reproducible run

Requirements: Python 3.10+ (the recorded run used the exact versions in `requirements.txt`), internet access for the initial model and dataset downloads, and a CPU or CUDA-capable PyTorch installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_ner.py --config config.json
```

The default run uses CPU, seed 42, the pinned model and dataset revisions in `config.json`, and the official train/validation/test splits. The best validation-F1 checkpoint is retained, then evaluated on validation and test. Run artifacts (checkpoints, model weights, logs, exact runtime metrics, and resolved config) are written under `outputs/` and are intentionally not tracked by Git. Use a fresh `--output-dir` for a new run if preserving previous artifacts.

`resolved_config.json` records exact input revisions, parameters, split sizes, runtime, and software versions. Re-running may produce small hardware/library-dependent differences; compare using the pinned versions and commit hashes.

## Implementation notes

- Uses a fast DistilBERT tokenizer with pre-tokenized words.
- The NER tag is applied to the first subword only. Continuation subwords and special tokens receive label `-100`, so they do not contribute to loss or metrics.
- Validation F1 is the checkpoint-selection metric. Precision/recall/F1 are entity-level scores from `seqeval`; token accuracy is also reported as a supplemental metric.
- The test split is not used for checkpoint selection.
- No dataset files or model weights are included in this repository.

## Data and model provenance

- Dataset: [TNER CoNLL-2003](https://huggingface.co/datasets/tner/conll2003), pinned to the commit in `config.json`; the script downloads its JSONL splits and label map from that exact revision without running its Python loader.
- Original task: Tjong Kim Sang and De Meulder, [Introduction to the CoNLL-2003 Shared Task](https://aclanthology.org/W03-0419/).
- The dataset card identifies the English source as Reuters newswire. The original task and dataset-card notices state that Reuters text is copyrighted and subject to Reuters/NIST agreements; the repository does not redistribute the data. Check the underlying terms before use beyond this demonstration.
- Base model: [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased), pinned to the revision in `config.json`; its model card lists Apache-2.0.
- The included MIT license covers only original code in this repository. It does not relicense the dataset or base-model files.


