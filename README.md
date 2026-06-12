# Content-Based Movie Recommendation System

This project is a **Content-Based Movie Recommendation System** built with Python and Machine Learning.

The main goal of this project is to recommend similar movies based on their textual content, especially the movie overview/description.

Unlike collaborative filtering, this system does not depend on user ratings. Instead, it recommends movies by analyzing movie features such as descriptions and finding movies with similar content.

## Project Overview

In this project, I used a movie dataset that contains information about movies, including:

* movie title
* original title
* overview
* release information
* rating-related information

The recommendation system uses the `overview` column to understand the content of each movie. Then, it finds movies with similar descriptions using TF-IDF and similarity calculation.

## Recommendation Type

This project uses:

```text
Content-Based Filtering
```

Content-Based Filtering recommends items based on their own features.

For example, if a movie is about space, science fiction, and adventure, the system will recommend other movies with similar descriptions.

## Dataset

The dataset contains movie metadata.
The most important column used in this project is:

```text
overview
```

This column contains the description or summary of each movie.

Another important column is:

```text
original_title
```

This column is used to search movies by their title and return recommendations.

## Data Preprocessing

First, missing values in the `overview` column were replaced with an empty string.

```python
movies_cleaned["overview"] = movies_cleaned["overview"].fillna("")
```

This step is important because text vectorizers cannot process missing values such as `NaN`.

## Text Vectorization with TF-IDF

To convert movie descriptions into numerical values, I used `TfidfVectorizer`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(
    min_df=3,
    max_features=None,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    stop_words='english'
)
```

### Why TF-IDF?

TF-IDF stands for:

```text
Term Frequency - Inverse Document Frequency
```

It gives importance to words that are meaningful in a movie description.

Common words like `the`, `is`, `and`, `of` are removed using English stop words.

Then, the overview text is transformed into a TF-IDF matrix:

```python
tfv_matrix = tfv.fit_transform(movies_cleaned["overview"])
```

Each movie is now represented as a numerical vector based on its description.

## Similarity Calculation

After converting text into numerical vectors, I used `sigmoid_kernel` to calculate similarity between movies.

```python
from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
```

This creates a similarity matrix where each movie is compared with every other movie.

For example:

```text
Movie A vs Movie B → similarity score
Movie A vs Movie C → similarity score
Movie A vs Movie D → similarity score
```

Movies with higher similarity scores are considered more similar.

## Creating Movie Index Mapping

To find a movie by its title, I created a mapping between movie titles and their DataFrame index.

```python
indices = pd.Series(
    movies_cleaned.index,
    index=movies_cleaned["original_title"]
).drop_duplicates()
```

This allows the system to quickly find the index of a movie.

Example:

```python
indices["Avatar"]
```

This returns the row index of the movie `Avatar`.

## Recommendation Function

The recommendation function takes a movie title as input and returns similar movies.

```python
def give_rec(title, sig=sig):
    idx = indices[title]

    sig_scores = list(enumerate(sig[idx]))

    sig_scores = sorted(
        sig_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sig_scores = sig_scores[1:11]

    movie_indices = [i[0] for i in sig_scores]

    return movies_cleaned["original_title"].iloc[movie_indices]
```

### How the Function Works

1. It finds the index of the selected movie.
2. It gets similarity scores between this movie and all other movies.
3. It sorts movies by similarity score.
4. It removes the first result because it is usually the same movie.
5. It returns the top 10 most similar movies.

## Example Usage

```python
give_rec("Avatar")
```

Example output:

```text
1. John Carter
2. Star Trek Into Darkness
3. The Avengers
4. Guardians of the Galaxy
5. Aliens
```

The output depends on the dataset and similarity scores.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Sigmoid Kernel
* Jupyter Notebook

## Machine Learning Concepts Used

* Content-Based Filtering
* Natural Language Processing
* TF-IDF
* Text Vectorization
* Similarity Matrix
* Sigmoid Kernel
* Movie Recommendation System

## Difference from Collaborative Filtering

Collaborative filtering recommends movies based on user behavior and ratings.

Content-based filtering recommends movies based on movie features, such as:

* overview
* genre
* keywords
* cast
* director

In this project, recommendations are mainly based on movie descriptions.

## Project Goal

The goal of this project is to understand how content-based recommendation systems work and how text data can be used to recommend similar movies.

This project helped me practice:

* text preprocessing
* TF-IDF vectorization
* similarity calculation
* recommendation system logic
* working with movie metadata
