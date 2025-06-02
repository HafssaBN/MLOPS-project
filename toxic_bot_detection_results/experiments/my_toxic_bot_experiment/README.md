# 🤖 Toxic/Bot Detection Results - my_toxic_bot_experiment

## 📊 Experiment Summary
- **Timestamp**: 2025-06-02 14:17:29
- **Best Validation Accuracy**: 0.9150
- **Total Trials Completed**: 20
- **Platform**: Google Colab

## 📈 Dataset Statistics
- **Total Comments**: 26,166
- **Toxic/Bot Comments**: 22,969 (87.78%)
- **Normal Comments**: 3,197

## 🏆 Best Model Configuration
```json
{
  "architecture": "ImprovedResNet",
  "hidden_dim": 256,
  "dropout_rate": 0.10301892772651725,
  "batch_size": 128,
  "learning_rate": 0.006191801342424958,
  "optimizer": "AdamW",
  "weight_decay": 0.0001640148320264181,
  "num_blocks": 1
}
```

## 📁 Files in This Experiment
- `study_results.json` - Main experiment results
- `top_trials.json` - Best performing trials
- `tokenizer.pkl` - Text preprocessing tokenizer
- `model_config.json` - Model configuration and weights
- `data_summary.json` - Dataset statistics and samples
- `quick_usage.py` - Ready-to-use code snippets

## 🚀 Quick Start in New Colab Session
```python
# 1. Load results
import json
with open('study_results.json', 'r') as f:
    results = json.load(f)

# 2. Load tokenizer
import pickle
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# 3. Check best parameters
print("Best model params:", results['best_params'])
print("Best accuracy:", results['best_value'])
```

## 💾 Google Drive Location
If saved to Drive: `/content/drive/MyDrive/toxic_bot_detection_results/experiments/my_toxic_bot_experiment/`

---
*Generated on Google Colab - 2025-06-02 14:17:29*
