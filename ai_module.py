from openai import OpenAI
import numpy as np

base_url = "https://api.aimlapi.com/v1"
api_key = "6acc884872534b729efbeecf46954564"
# system_prompt = "You are a travel agent. Be descriptive and helpful."
# user_prompt = "Tell me about San Francisco"

api = OpenAI(api_key=api_key, base_url=base_url)


# def main():
# def analyze(past_predictions, new_predictions):
#     # past_list = list(past_predictions)
#     # new_list = list(new_predictions)
    
#     # user_prompt = f"Past Historical Data: {past_list}, Future Predictions: {new_list}"
#     user_prompt = f"{past_predictions}, {new_predictions}"
#     completion = api.chat.completions.create(
#         model="mistralai/Mistral-7B-Instruct-v0.2",
#         messages=[
#             {"role": "system", "content": ""},
#             {"role": "system", "content": ""},

#             # {"role": "user", "content": user_prompt},
#             {"role": "user", "content": user_prompt},

#         ],
#         temperature=0.7,
#         max_tokens=256,
#     )

#     response = completion.choices[0].message.content

#     # print("User:", user_prompt)
#     print("AI:", response)


# def analyze(past_slope, predicted_slope, threshold=0.1):
#     """
#     Analyzes whether the predicted slope is within a reasonable range of the past slope.
    
#     Args:
#         past_slope (float): Magnitude of the slope from historical stock data.
#         predicted_slope (float): Magnitude of the slope from predicted stock data.
#         threshold (float): Maximum allowed difference between the slopes for them to be considered "reasonable".
        
#     Returns:
#         int: 1 if the predicted slope is within the reasonable range, 0 otherwise.
#     """
#     # Calculate the absolute difference between the slopes
#     difference = abs(past_slope - predicted_slope)
    
#     # Check if the difference is within the threshold
#     if difference <= threshold:
#         return 1
#     else:
#         return 0

# print(analyze(1.4, 1.49))
# if __name__ == "__main__":
#     main()


def analyze_with_ai(past_slope, predicted_slope):
    """
    Uses the AI model to analyze whether the predicted slope is within a reasonable range of the past slope.
    
    Args:
        past_slope (float): Magnitude of the slope from historical stock data.
        predicted_slope (float): Magnitude of the slope from predicted stock data.
        
    Returns:
        int: 1 if the predicted slope is within a reasonable range, 0 otherwise.
    """
    user_prompt = (
        f"Past slope: {past_slope}, Predicted slope: {predicted_slope}. "
        "Return 1 if reasonable, else 0. DO NOT RETURN WORDS"
    )
    
    # try:
    completion = api.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=5,
    )
    response = completion.choices[0].message.content.strip()
    print(response)
    return int(response) if response in ("0", "1") else -1
    # except api.BadRequestError as e:
    #     print(f"API error: {e}")
    #     return -1  # Fallback result

# Example usage
result = analyze_with_ai(0.25, 0.28)
print(result)
