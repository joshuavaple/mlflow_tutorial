from collections import Counter
import os
import datetime

# helper functions
# Step 1: Find the highest value key within each sub-dictionary
def get_popular_vote_with_error_handling(data: dict):
    highest_keys = {}
    try:
        for key, sub_dict in data.items():
            highest_key = max(sub_dict, key=sub_dict.get)
            highest_keys[key] = highest_key
        # Step 2: Perform a voting process to determine the key with the highest value across all sub-dictionaries
        votes = Counter(highest_keys.values())
        highest_voted_key = votes.most_common(1)[0][0]
        return highest_voted_key
    except Exception as e:
        print("Errored during summarizing step, returned empty string")
        print(e)
        return ""


def zero_shot_predict_single_model(classifier, sequence_to_classify: str, candidate_labels: list):
    try:
        predictions = classifier(sequence_to_classify, candidate_labels)
        result = {predictions['labels'][i]: predictions['scores'][i] for i in range(len(predictions['labels']))}
        return result
    except Exception as e:
        print("The following error occured, returned empty string")
        print(e)
        return {}


def zero_shot_predict_multiple_models(classifiers: dict, sequence_to_classify: str, candidate_labels: list):
    result_dict = {}
    try:
        for classifier_name in classifiers:
            classifier = classifiers[classifier_name]    
            result = zero_shot_predict_single_model(classifier, sequence_to_classify, candidate_labels)
            result_dict[classifier_name] = result
        return result_dict
    except Exception as e:
        print("The following error occured, returned empty string")
        print(e)
        return {}

def summarize_text(summarization_pipeline, input_text, max_char:int):
    try:
        return summarization_pipeline(input_text[:max_char], max_length=200, min_length=100, do_sample=False)[0]['summary_text']
    except Exception as e:
        print("The following error occured, returned empty string")
        print(e)
        return ""
    
def get_top_n_label_and_score(data_dict: dict, top_n):
    """
    This function taks in a dictionary where each key's value is a number (score) and return the nth highest key
    """
    # check if all values are numbers:
    values = [data_dict[key] for key in data_dict.keys()]
    check = all([isinstance(item, (float,int)) for item in values])
    if check:
        try:
            # Sort the dictionary items by values in descending order
            sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True) # returns a list of (key, value) tuples
            # Get the key and value for top_n value
            top_key = sorted_items[top_n-1][0]
            top_value = sorted_items[top_n-1][1]
            return top_key, top_value
        except Exception as e:
            print('The following error occurs when processing the input data, an empty string and zero were returned')
            print(e)
            return "",0
    else:
        print("values in input data contain non-numberical elements, an empty string and zero were returned")
        return "",0

def save_dataframe_with_timestamp(prefix, destination_path, dataframe, file_extension=".csv"):
    try:
        # Ensure the destination directory exists
        if not os.path.exists(destination_path):
            print("The destination specified does not exist, creating")
            os.makedirs(destination_path)
            print(destination_path)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{prefix}_{current_datetime}{file_extension}"
        destination_file_path = os.path.join(destination_path, filename)
        dataframe.to_csv(destination_file_path, index=False)
        print(f"Saved DataFrame as '{filename}' in the destination directory.")
    except Exception as e:
        print(f"Error: {e}")