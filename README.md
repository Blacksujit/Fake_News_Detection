# Fake News Detector

This project aims to detect fake news using machine learning. The project includes a Flask-based application that serves both backend (for model training and prediction) and frontend (for user interaction).

# Web Page :

![alt text](image.png)

## Folder Structure

- `app.py`: Main application file combining both frontend and backend.
- `model/`: Contains the trained model and script to train the model.
- `requirements.txt`: Dependencies for the project.
- `static/`: Static files (CSS, JS).
- `templates/`: HTML templates.
- `data/`: Folder containing the dataset.

## Dataset Information:

Download the dataset from [Kaggle Machine Learning](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

## Installation

1. Clone the repository.

2. Download a dataset suitable for fake news detection and place the file in the `data/` folder (e.g., `fakenews.csv`).

3. Virtual Environment setup
    
    ```bash
    python -m venv venv
    ```
    
    ```bash
    venv\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python run.py
    ```
6. Open your browser and go to `http://127.0.0.1:5000` to use the application.

## License

This project is licensed under the MIT License.
