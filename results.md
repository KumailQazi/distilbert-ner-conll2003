# Results

**Run status:** completed successfully on 2026-10-05. These are measured results from the run recorded in `resolved_config.json`; they are not estimates.

## Evaluation

The reported precision, recall, and F1 are entity-level `seqeval` scores using first-subword labels. Token accuracy is included as a supplemental metric. The best checkpoint was selected by validation F1 (epoch 3); the test split was evaluated after selection and was not used for model selection.

| Split | Loss | Precision | Recall | F1 | Token accuracy |
|---|---:|---:|---:|---:|---:|
| Validation (3,250 sentences) | 0.05734 | 0.93850 | 0.94198 | **0.94024** | 0.98725 |
| Test (3,453 sentences) | 0.14254 | 0.89458 | 0.90204 | **0.89829** | 0.97885 |

Training used 14,041 sentences for 3 epochs (5,268 optimizer steps). Training loss was 0.04483.

## Hardware and runtime

- Device: CPU only; CUDA unavailable
- CPU reported by the environment: Intel Xeon Processor @ 2.50 GHz, 8 logical CPUs
- System memory: 23.64 GiB; measured peak process RSS: 1,972.2 MiB
- Training time: 2,775.248 seconds (**46 min 15 sec**)
- End-to-end run time including final validation/test and saving: 2,835.090 seconds (**47 min 15 sec**)
- Software: Python 3.12.3, PyTorch 2.5.1+cpu, Transformers 4.47.1, Datasets 3.2.0, SeqEval 1.2.2, Accelerate 1.2.1; NumPy 1.26.4 and PyArrow 18.1.0 are pinned in `requirements.txt`.

This was run on the available Linux CPU environment.Its runtime may differ with CPU, thermals, software builds, and background load.

## Interpretation and limits

This demonstrates a reproducible DistilBERT token-classification fine-tune on CoNLL-2003. It is general sequence-labeling evidence only. It is **not** an Arabic diacritization model, does not reproduce the specific character-level CATT architecture, does not report diacritic/word error rates, and does not establish a structural guarantee that underlying letters cannot change. Do not use these NER metrics as a proxy for other diacritization performance.

The repository contains no dataset files or trained weights. The source dataset is Reuters-derived; review the data terms noted in `README.md` before any use or redistribution beyond this demonstration.
