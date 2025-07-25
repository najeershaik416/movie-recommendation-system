
# 🎬 CineSuggest: An AI-Powered Movie Recommendation System

CineSuggest is an interactive web application built with Streamlit that provides personalized movie recommendations. This project evolved from a data science experiment in a Kaggle notebook into a full-fledged, deployable application. It leverages a variety of recommendation techniques, from classic content-based filtering to modern AI-powered natural language search.



-----

##  ✨ Features

CineSuggest offers five distinct ways to discover movies, catering to different user needs:

1.  *🌟 For First-Time Visitors (Popular Movies):* New to the app? Get a list of the most popular and highly-rated movies based on thousands of user ratings.
2.  *👤 For Registered Users (History-Based):* Enter your User ID to get recommendations tailored to your unique taste, based on movies you've previously watched and loved.
3.  *🔧 Custom Filter (Rule-Based):* Fine-tune your search with specific rules. Filter movies by release year and exclude genres you don't like.
4.  *🔍 Find Similar Movies (Content-Based):* Pick a movie you love, and the system will find others with similar genres and themes using cosine similarity.
5.  *🤖 AI-Powered Search (Natural Language):* The most advanced feature. Simply type what you're looking for in plain English (e.g., "a mind-bending sci-fi movie like Inception but with more action"), and our AI backend will parse your request and find the perfect movies for you\!

-----

## 🛠 Tech Stack & Data

  * *Language:* Python 3.9+
  * *Core Libraries:* Streamlit, Pandas, Scikit-learn, Requests
  * *Dataset:* [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/)
  * *External API:* DeepSeek model via OpenRouter for natural language processing.

-----

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

  * Python 3.9 or higher
  * Git

### 1. Clone the Repository

bash
git clone (https://github.com/najeershaik416/movie-recommendation-system?tab=readme-ov-file)


### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


###  3. Install Dependencies

Install all the required Python libraries from the requirements.txt file.

bash
pip install -r requirements.txt


###  4. Configure API Key

The AI-Powered Search requires an API key from a service like OpenRouter.

1.  Create a folder named .streamlit in the root of the project directory.

2.  Inside this folder, create a file named secrets.toml.

3.  Add your API key to this file as shown below:

    toml
    # .streamlit/secrets.toml
    DEEPSEEK_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    

###  5. Configure Data Source

This application loads the MovieLens dataset directly from Google Drive to avoid issues with large file sizes in the repository.

1.  Upload your movie.csv and rating.csv files to your own Google Drive.
2.  Get a shareable link for each file (ensure it's set to "Anyone with the link").
3.  Convert these links to direct download URLs.
4.  Open the app.py file and update the MOVIES_URL and RATINGS_URL variables with your direct download links.

-----

##  ▶ How to Run the App

Once the setup is complete, you can run the Streamlit application with a single command:

bash
streamlit run app.py




-----

##  📂 Project Structure


.
├── .streamlit/
│   └── secrets.toml      # Securely stores your API key
├── app.py                # The main Streamlit application file (UI)
├── recommender.py        # The core recommendation engine and logic
├── requirements.txt      # A list of all Python dependencies
└── README.md             # You are here!


-----

##  📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

-----

##  🙏 Acknowledgements

This project uses the MovieLens 20M dataset, collected and maintained by the [GroupLens research group](https://grouplens.org/) at the University of Minnesota.
