# Empathy Predicition from chat responses

This project attempts to predict empathy score from an AI-generated response using machine learning and natural language processing techniques. The work is based on a dataset from a preregistered five-study experiment examining how humans evaluate empathy differently when responses are perceived as AI-generated versus human-written.


## Data
Our data contains $\approx 3.5K$ samples from different studies.  It consists of chat conversations between participants and GPT (model: GPT 4-0613). Each record corresponds to an AI response within a conversation and includes only the AI response text (the participant’s stories are not included).
The original [study](https://doi.org/10.1038/s41562-025-02247-w) examines whether humans evaluate empathy differently for responses perceived as AI-generated or written by humans. While manipulating the responses’ perceived source to be written by humans or AI,
the examiners measured the participants' perception of empathy in their responses, as well as other empathy measurements (positive/negative emotions, and whether the participants felt supported following the response). One of the main findings of this study is that telling people a reply is from a human boosts empathy and related judgments. 
The data includes:
**Textual features**: AI response text (participant stories not included)
**Metadata**: Response source perception (AI vs human), participant AI attitudes, and empathy ratings (0-9 scale)

The data is highly imbalanced: most samples have high empathy scores.
The exploratory data analysis can be found in `exploratory_data_analysis` path, and the visualization of the knowledge graph can be found in [here](https://morbea.github.io/EmpathyPrediction/).

## Goals:

1. Explore patterns in empathy scores through data analysis
2. Build baseline ML models for empathy prediction.
3. Use advanced NLP techniques for empathy prediction.


## Methods Tested

### Baseline Models
- Random Forest + XGBoost with TF-IDF and SBERT embeddings
- Apply class imbalance methods (such as SMOTE) to handle the class imbalance

### Advanced Models
- LSTM networks
- Fine-tuned BERT
- Few shot learning
- Implementation of the HCBOU algorithm to handle the imabalnce in the labels.


## Results

All models struggled due to severe class imbalance and dataset size limitations:
- Basic machine learning models: Poor minority class prediction despite weighting
- LSTM: showed a slight improvement with SBERT embeddings, but insufficient.
- Fine-tuned BERT: although it showed a minor improvement, it failed on simplified binary classification
- Few-shot learning: The predictions were biased toward majority class predictions

## Project Structure

```
├── README.md                           
├── index.html                        # graph visualization
├── data/                             # Dataset files (hidden)
├── exploratory_data_analysis/        # EDA notebooks
│   ├── exploratory_analysis.ipynb    # data exploration
│   └── knowledge_graph.ipynb         # Knowledge graph visualization code
└── prediction/                       # model prediction notebooks
    ├── Fine_tuning_empathy.ipynb     # BERT fine-tuning
    ├── HCBOU_implementation.ipynb    # implementation of the HCBOU algorithm
    ├── few_shot_learning.ipynb       # Few-shot learning
    └── prediction_models.ipynb       # Prediction using ML and deep learning models
```