# 📘 Descriptive Overview: BERT Question Answering System

**A Colab-based deep learning project that fine-tunes BERT on the SQuAD dataset to answer questions from context paragraphs.**

This project demonstrates how to build a **BERT-based Question Answering (QA) model** from scratch using the **SQuAD v1.1 dataset**. The core objective is to enable machines to answer natural language questions based on a given paragraph or context. It leverages **transfer learning** with the `bert-base-uncased` model provided by Hugging Face.

The QA pipeline involves:

* Loading the dataset
* Preprocessing it to match BERT's format
* Fine-tuning BERT on the QA task
* Predicting answers from context

This notebook is designed to run on **Google Colab**, supports **GPU acceleration**, and stores results on **Google Drive**.

## 📂 Project Structure

```
BERT_QA_System/
├── squad_train.csv                # Exported training data (SQuAD)
├── squad_validation.csv           # Exported validation data
├── tokenized_squad_train/         # Tokenized training dataset (HF format)
├── tokenized_squad_val/           # Tokenized validation dataset (HF format)
├── bert-qa/                       # Output directory for the trained model
├── logs/                          # Training logs
├── QA_script.ipynb                # Main Colab notebook with code
└── README.md                      # This file
```

## 🚀 Features

* ✅ Load and preprocess the SQuAD dataset
* ✅ Tokenize using `BertTokenizerFast`
* ✅ Fine-tune `BertForQuestionAnswering`
* ✅ Predict answers from context paragraphs
* ✅ Save and reload tokenized datasets
* ✅ Optional integration with Weights & Biases for tracking

## 🛠 Requirements

* Python 3.7+
* Transformers ≥ 4.28
* Datasets ≥ 2.10
* Torch ≥ 1.12
* Google Colab (preferred)

## 🧪 Usage Instructions

1. Open the notebook in Google Colab.
2. Enable GPU: **Runtime > Change runtime type > GPU**.
3. Run each cell in order.
4. Mount Google Drive to save/export your model and datasets.
5. (Optional) Reduce dataset size for faster testing.
6. Train the model for 1 epoch.
7. Use `predict_answer()` to infer answers.

## 🧠 Model Description

* We use `bert-base-uncased`, a transformer model pretrained on a large corpus of English data.
* It is fine-tuned on the SQuAD dataset, which consists of questions and answer spans within context paragraphs.
* The model learns to predict the start and end token positions of the answer in the context.

## 🔧 Customization

* ✏️ Replace default question/context with your own for testing.
* 💾 Save the model with `model.save_pretrained()`.
* 🔁 Reload later with `from_pretrained()`.

## 📈 Optional: Tracking with Weights & Biases

Enable experiment tracking by:

```python
import wandb
wandb.login()  # You’ll be prompted for API key
```

To disable logging:

```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

## 📚 References

* [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
* [Google Colab](https://colab.research.google.com)

---

**Author:** Aditya Verma 
**Date:** July 2025
