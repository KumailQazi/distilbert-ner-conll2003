# Local Test Run Results

The dataset was split into 14,041 training sentences, 3,250 validation sentences, and 3,453 test sentences. “Training” teaches the model; “validation” helps select the best checkpoint; the “test” split is kept separate until final scoring. An “epoch” is one complete pass through the training examples. The run did 3 epochs, batch size 8 (eight examples processed at a time), maximum input length 128 tokens, learning rate 0.00005, and random seed 42. The seed helps make random choices repeatable; different hardware can still produce small differences.

Validation results were precision 0.93850 (93.850%), recall 0.94198 (94.198%), F1 0.94024 (94.024%), loss 0.05745, and token accuracy 0.98730 (98.730%). Test results were precision 0.89499 (89.499%), recall 0.90293 (90.293%), F1 0.89894 (89.894%), loss 0.14236, and token accuracy 0.97893 (97.893%).

Precision asks, “Of the entities the model predicted, how many were correct?” Recall asks, “Of the real entities, how many did it find?” F1 combines precision and recall into one score; it is not the same as accuracy. Loss is a training/evaluation error measure, not a percentage; lower is generally better when comparing the same task and setup. Token accuracy counts individual labeled word positions and can look high even when an entity span is partly wrong, so entity-level F1 is more useful for NER.

The best checkpoint was selected using validation F1; the test split was evaluated afterward. Training took 2 hours 33 minutes 21 seconds (9,201.289 seconds); the full run including evaluation and saving took 2 hours 37 minutes 35 seconds (9,455.525 seconds). It ran CPU-only on a 4-logical-CPU Intel Core i7 Windows environment with zero CUDA GPU acceleration. CPU is a general-purpose processor; GPU is specialized for parallel math; CUDA is NVIDIA’s GPU-computing platform. No GPU run or MacBook runtime was measured for this NER project, and the report does not give NER words-per-second.

## Measured Comparison: Baseline (Linux Xeon) vs. Re-run (Local Windows CPU)

| Metric / Attribute | Baseline Run (`results.md`) | Re-run (`outputs/test_run`) | Delta / Notes |
| :--- | :---: | :---: | :--- |
| **Validation Loss** | 0.05734 | **0.05745** | +0.00011 |
| **Validation Precision** | 0.93850 | **0.93850** | 0.00000 |
| **Validation Recall** | 0.94198 | **0.94198** | 0.00000 |
| **Validation Entity F1** | **0.94024** | **0.94024** | **0.00000** |
| **Validation Token Acc.** | 0.98725 | **0.98730** | +0.00005 |
| **Test Loss** | 0.14254 | **0.14236** | -0.00018 |
| **Test Precision** | 0.89458 | **0.89499** | +0.00041 |
| **Test Recall** | 0.90204 | **0.90293** | +0.00089 |
| **Test Entity F1** | **0.89829** | **0.89894** | **+0.00065** |
| **Test Token Acc.** | 0.97885 | **0.97893** | +0.00008 |
| **Training Time** | 46m 15s | **2h 33m 21s** | 8 vCPU Xeon server vs. 2-core / 4-thread laptop CPU |
| **Environment** | Linux, 8 vCPUs | Windows 10, 4 vCPUs | PyTorch 2.5.1+cpu |

## Artifact Locations & Model Weights

- **Output Directory:** `outputs/test_run/`
- **Trained Model Weights:** `outputs/test_run/best_model/`
- **Execution Metrics:** `outputs/test_run/metrics.json`
- **Resolved Configuration:** `outputs/test_run/resolved_config.json`
- **Task Log:** `task-185.log`
