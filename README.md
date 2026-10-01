# SHL Hiring Assessment 2026: grammar scoring from speech

This is my solution to the SHL Hiring Assessment 2026 Kaggle competition. The task is to predict a 0–5 grammar score for spoken English clips (769 train, 216 test). The metric is RMSE.

| | RMSE |
|---|---|
| Public leaderboard | **0.3382** |
| Speaker-grouped cross-validation | 0.509 |
| Training (in-sample) | 0.225 |

## Files

- **`extract_features.py`** turns the audio into features, written to `features/`. It needs GPUs.
- **`shl_grammar_scoring.ipynb`** holds the modelling and the report:
  - validation design;
  - base models and stacking;
  - evaluation and plots;
  - training RMSE;
  - the submission.

## Approach

1. **Speaker-grouped validation.**
   - Training candidates recorded about three answers each, with near-identical grades. Test speakers never appear in training.
   - Random K-fold therefore leaks speaker identity.
   - All model selection uses folds grouped by speaker. The speaker groups are inferred from voice similarity.
2. **Grade-0 gate.** The 37 clips graded 0 are a separate, uniformly loud batch. A speech-ratio threshold catches them exactly.
3. **Representations.** Each clip is described by frozen pretrained models:
   - Whisper large-v3 encoder and WavLM-Large;
   - the Whisper decoder fed the ASR transcripts;
   - Qwen2-Audio-7B and Voxtral-Mini-3B, listening with the rubric;
   - Qwen2.5-7B and 14B, reading transcripts from Whisper and Parakeet against the rubric;
   - interpretable fluency and grammar features, such as ASR confidence, pauses, syntax and grammatical-error edit rate.
4. **Model.**
   - 17 Ridge, SVR and LightGBM regressors.
   - They are combined by non-negative least squares, with separate weights for the ~45 s recording batch that dominates the test set.
   - Predictions are clipped to [2, 5].

## Running

**1. Feature extraction**

Run on a machine with two GPUs. We used Kaggle (2× T4, internet on, competition data attached), about 8 GPU-hours in total.

```
pip install faster-whisper==1.1.1 "nemo_toolkit[asr]" "bitsandbytes>=0.46.1" "mistral-common[audio]" spacy
python -m spacy download en_core_web_sm
python extract_features.py
```

The steps can also be run separately, for example `--steps text_llm gec`. Use `--n-clips 4` for a quick test.

**2. Modelling**

Runs on a CPU in about a minute. Put the competition `train.csv` and `test.csv` in `data/` and the extracted features in `features/`, then:

```
pip install numpy pandas scipy scikit-learn lightgbm matplotlib jupyter
jupyter nbconvert --execute --to notebook shl_grammar_scoring.ipynb
```

The notebook writes `submission.csv`.

The competition data, the extracted features and the submission file are not included in this repository; the competition rules forbid sharing the data. No external data was used.
