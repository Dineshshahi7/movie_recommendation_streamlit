# Movie Recommendation System (Content-Based Filtering)

## Project Overview
This project is a **content-based movie recommendation system** built using Python and Streamlit.  
It recommends movies to users based on **similarity between movie content and user preferences**, rather than using machine learning models.

The system analyzes movie features and calculates similarity scores to suggest movies that are most relevant to the user’s selected movie.

---

## Problem Statement
With a large number of movies available on streaming platforms, users often struggle to find movies that match their interests.

Key challenges include:
- Difficulty in discovering relevant movies  
- Overwhelming number of choices  
- Lack of personalized recommendations  

This project aims to solve these challenges by recommending **similar movies based on content similarity**, helping users quickly discover movies aligned with their preferences.

---

## Solution Approach
The recommendation system follows a **content-based filtering approach**, which works as follows:

1. Analyze movie attributes such as genre, description, keywords, or tags  
2. Convert movie information into numerical vectors  
3. Calculate similarity between movies using similarity measures  
4. Recommend movies that are most similar to the user’s selected movie  

This approach ensures recommendations are based purely on **movie content**, not user ratings or historical behavior.

---

## Tools & Technologies Used
- Python  
- Pandas & NumPy  
- Scikit-learn (for similarity calculation)  
- Streamlit  
- Jupyter Notebook  

---

## Similarity Technique Used
- **Cosine Similarity**

Cosine similarity is used to measure how similar two movies are based on their content features.  
Movies with higher similarity scores are recommended to the user.

---

## Dataset Description
The dataset contains information about movies, including:
- Movie titles  
- Genres  
- Descriptions / tags / keywords  

The dataset is preprocessed to remove missing values and prepare the data for similarity calculation.

---

## Application Features
- User can select a movie from the list  
- System finds movies with similar content  
- Displays a list of recommended movies  
- Simple and interactive user interface using Streamlit  

---

## How to Run the Project

### Live Deployment (Streamlit App)
This project is deployed as an interactive web application using Streamlit.

🔗 **Live Demo:**  
https://movierecommendationapp-07.streamlit.app/

Users can select a movie and instantly receive recommendations for similar movies.

---

### 🧪 Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Dineshshahi7/movie_recommendation_streamlit.git
