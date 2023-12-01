# Sentiment Classification Model Training with BERT

### Sentiment Classification Model Training Guide with BERT

This guide outlines the necessary steps to train a sentiment classification model using BERT (Bidirectional Encoder Representations from Transformers). It utilizes a Python and PyTorch setup, and the model is trained on a dataset containing Spanish opinions.

### Environment Setup

1. **Initial Imports:**
   - Make sure to import all necessary libraries before starting the training.

   ```python
   import multiprocessing
   import torch
   import pandas as pd
   from transformers import BertTokenizer, BertForSequenceClassification
   # ...other imports...
   
   ```

2. **Multiprocessing and CUDA Setup:**
   - Set the multiprocessing start method and check for CUDA availability to use the GPU.

   ```python
   multiprocessing.set_start_method('spawn', True)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f'Using device: {device}')
   
   ```

### Data Preparation

1. **Data Loading and Cleaning:**
   - Load data from a CSV file and perform initial cleaning, such as splitting long texts.

   ```python
   df = pd.read_csv('combined.csv')
   df = split_long_texts(df)
   
   ```

2. **Tokenization and Dataset:**
   - Use `BertTokenizer` to tokenize the data and prepare a custom PyTorch dataset.

   ```python
   tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
   dataset = CustomDataset(df, tokenizer, MAX_LEN)
   
   ```

### Model Configuration

1. **Loading Pretrained Model:**
   - Load BERT with the necessary configuration for sequence classification.

   ```python
   model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
   model.to(device)
   
   ```

2. **Optimizer and Scheduler:**
   - Set up the optimizer and scheduler for model training.

   ```python
   optimizer = AdamW(model.parameters(), lr=2e-5)
   scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(df)*3)
   
   ```

### Training Process

1. **Training by Epochs:**
   - Train the model for several epochs, recording and adjusting parameters as needed.

   ```python
   for epoch in range(EPOCHS):
       # Training and validation...
       print(f"Epoch: {epoch+1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
   
   ```

2. **Evaluation and Fine-tuning:**
   - Evaluate the model with the validation set to determine accuracy.

   ```python
   train_accuracy, train_loss = train_epoch(...)
   val_accuracy, val_loss = eval_model(...)
   ```

### Prediction and Evaluation

1. **Prediction Function:**
   - Develop a function to predict sentiments of new texts.

   ```python
   predictions = predict_sentiments(test_texts, model, tokenizer, MAX_LEN)
   
   ```

2. **Results and Refinement:**
   - Interpret the results, fine-tune the model if necessary, and make predictions on unseen data.

   ```python
   sentiments, probabilities = predict_sentiments(test_texts, test_targets, model, tokenizer, MAX_LEN)
   
   ```

### Conclusion and Next Steps

- Analyze the results to understand the strengths and weaknesses of the model.
- Consider further training with additional data or adjusting model parameters to improve accuracy.
- Upon completing training, save this model and its tokenizer for future use in the SoftwareBERTHealthCMV application.
