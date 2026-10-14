import requests # Import the requests library to handle HTTP requests
import json

def emotion_detector(text_to_analyse): # Define a function named emotion detector that takes a string input (text_to_analyse)
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict' # URL of the emotion predict service
    myobj = { "raw_document": { "text": text_to_analyse } } # Create a dictionary with the text to be analyzed
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"} # Set the headers required for the API request
    try:
        response = requests.post(url, json = myobj, headers=header, timeout=5)
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
        return {
            'anger': None, 'disgust': None, 'fear': None, 'joy': None,
            'sadness': None, 'dominant_emotion': None
        }

    if response.status_code == 400:
        return {
            'anger': None, 'disgust': None, 'fear': None, 'joy': None,
            'sadness': None, 'dominant_emotion': None
        }

    formatted_response = json.loads(response.text)
    if 'emotionPredictions' not in formatted_response:
        return {
            'anger': None, 'disgust': None, 'fear': None, 'joy': None,
            'sadness': None, 'dominant_emotion': None
        }
    
    # Extract the first prediction block from the emotionPredictions list
    emotion_predictions = formatted_response.get('emotionPredictions', [{}])[0]

    # Extract the inner emotion dictionary
    emotions = emotion_predictions.get('emotion', {})

    # Safely extract each individual score with a fallback default of 0
    anger_score = emotions.get('anger', 0)
    disgust_score = emotions.get('disgust', 0)
    fear_score = emotions.get('fear', 0)
    joy_score = emotions.get('joy', 0)
    sadness_score = emotions.get('sadness', 0)

    dominant_emotion = max(emotions, key=emotions.get)

    return {'anger': anger_score,'disgust': disgust_score,'fear': fear_score,'joy': joy_score,'sadness': sadness_score,'dominant_emotion': dominant_emotion}