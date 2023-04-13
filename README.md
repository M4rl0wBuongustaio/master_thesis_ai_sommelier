# AI-Sommelier (Master Thesis)

*AI-Sommelier installation instructions can be found below. (Installation Notes)*

## About this Repository and Research Questions

This repository documents the research process of a 
master thesis at the Julius-Maximilians-University of Würzburg. 
The thesis examined two main research questions:

- **RQ1.** Can a wine-recommendation system mainly based on the Transformer architecture outperform a conventional approach?
- **RQ2.** Is a Transformer model capable of adequately representing customers preferences, based on wine reviews and the common "wine-language" used in it, in the context of recommender systems?

### RQ1.

For answering **RQ1.** two collaborative filtering (CF) approaches have been proposed. 
The first (```collaborative_filtering_numeric.py```) identifies similar users based on their numeric ratings whereas
the second approach (```collaborative_filtering_textual.py```) uses a ```sentence_transformers``` model for identifying
similar users based on their review texts (<a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">URL to model</a>).

#### Results (RQ1.)

In three quantitative evaluation scenarios the Transformer-based 
approach could not <u>significantly</u> outperform the numeric approach (see exemplified results of two scenarios below).

| Scenario   | Type        | MSE        | RMSE       |
|------------|-------------|------------|------------|
| Base       | numeric     | **0.5233** | **0.7234** |
| Base       | transformer | 0.6042     | 0.7773     |
| cold-start | numeric     | 0.4071     | 0.6380     |
| cold-start | transformer | **0.4066** | **0.6377** |
 
### RQ2.

As the results for **RQ1.** illustrate the transformer's performance did not convince int the quantitative evaluation.
However, it demonstrated its capability of making sense out of textual wine-reviews. Therefore, 
the model was used to build an interactive recommendation app with ```streamlit```. The app takes a wine
description as an input and recommends *n* matching wines to this description. To test its performance, the app has been evaluated
in a qualitative interview with a professional sommelier (find results below).

#### Results (RQ2.)

| Match  | Creativity |
|:------:|:----------:|
| 64.75% |   57.5%    |

## Installation Notes

For using the AI-Sommelier please follow the subsequent steps:
1. clone this repository
2. install packages from ```requirements.txt```
3. download AI-sommelier data <u>and</u> model set as described in the Master Thesis (*Anhang*)
4. adjust ``EMBEDDING_PATH`` <u>and</u> ``EMBEDDER_PATH`` in ```ai_sommelier_backend.py``` accordingly
5. run ```ai_sommelier_frontend.py``` (e. g. from terminal)