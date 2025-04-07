# Project Overview

This project performs clustering on textual articles using sentence embeddings, dimensionality reduction, and clustering algorithms. It visualizes the results and outputs the articles grouped by clusters.
i use embedding model [nlpai-lab/KoE5](https://huggingface.co/nlpai-lab/KoE5)
## Features

- **Load Articles**: Reads articles from a text file.
- **Generate Embeddings**: Encodes articles using a SentenceTransformer model.
- **Dimensionality Reduction**: Reduces embedding dimensions with UMAP.
- **Clustering**: Clusters articles using HDBSCAN.
- **Visualization**: Visualizes clusters with PCA and Matplotlib.
- **Korean Font Support**: Configures font settings for Korean text display.

## Requirements

- Python 3.11.11
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. **Prepare Articles**: Add your articles to `example_articles.txt`, one per line.

2. **Set Font Path**: Create a `.env` file with the path to your Korean font:

   ```
   FONT_PATH=/path/to/your/font.ttf
   ```

3. **Run the Script**:

   ```bash
   python main.py
   ```

## Output

- Displays a scatter plot of the clustered articles.
- Prints cluster assignments for each article in the console.

## Notes

- Embeddings are saved in `embeddings.pkl` to speed up future runs.
- Ensure the font path is correct to properly display Korean characters.
> this readme is written by gpt-4o
