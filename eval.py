import openai
import os
import base64
import json
from datetime import datetime

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_type = "openai"

def encode_image_to_data_url(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{encoded_image}"
    return data_url

def get_puzzle_solution(image_path):
    image_data_url = encode_image_to_data_url(image_path)

    # Use OpenAI API to get a solution for the puzzle in the image
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Solve the word puzzle in the image. The answer is a common phrase, idiom, or saying."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting solution: {e}")
        return None

def evaluate_solution(solution, correct_answer):
    prompt = f"Is the following solution correct for the puzzle: '{solution}'? The correct answer is: '{correct_answer}'. Answer with 'Yes' or 'No'."

    # Use OpenAI API to evaluate the solution
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You help validate LLM created solutions"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )

        evaluation = response.choices[0].message.content
        return evaluation
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return None

def main():
    image_directory = 'data/images'
    text_directory = 'data/text'
    eval_directory = 'evals'

    if not os.path.exists(eval_directory):
        os.makedirs(eval_directory)

    results = []
    yes_count = 0
    no_count = 0

    # Get rid of any hidden files or otherwise
    image_files = sorted(f for f in os.listdir(image_directory) if f.endswith('.jpg') and not f.startswith('.'))
    text_files = sorted(f for f in os.listdir(text_directory) if f.endswith('.txt') and not f.startswith('.'))

    for i, (image_file, text_file) in enumerate(zip(image_files, text_files), start=1):
        image_path = os.path.join(image_directory, image_file)
        text_path = os.path.join(text_directory, text_file)

        if os.path.exists(image_path) and os.path.exists(text_path):
            with open(text_path, 'r') as file:
                correct_answer = file.read().strip()

            # Get the solution for the puzzle from the image
            solution = get_puzzle_solution(image_path)

            # Evaluate the solution
            evaluation = evaluate_solution(solution, correct_answer)
            print(f'Puzzle {i} Evaluation: {evaluation}')

            # Log the result
            result = {
                "puzzle_number": i,
                "solution": solution,
                "correct_answer": correct_answer,
                "evaluation": evaluation
            }
            results.append(result)

            # Count the evaluations
            if evaluation.lower() == "yes":
                yes_count += 1
            elif evaluation.lower() == "no":
                no_count += 1
        else:
            print(f'Puzzle {i} files are missing.')

    # Generate unique filename with date and time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_filename = f'{eval_directory}/eval_{timestamp}.json'

    # Save results to a JSON file
    with open(eval_filename, 'w') as eval_file:
        json.dump(results, eval_file, indent=4)

    # Print success metrics
    total_evaluations = yes_count + no_count
    if total_evaluations > 0:
        success_rate = (yes_count / total_evaluations) * 100
    else:
        success_rate = 0

    print('---- Evaluation Summary ----')
    print(f'Evaluation results saved to {eval_filename}')
    print(f'Total Evaluations: {total_evaluations}')
    print(f'Successful Evaluations (Yes): {yes_count}')
    print(f'Unsuccessful Evaluations (No): {no_count}')
    print(f'Success Rate: {success_rate:.2f}%')

if __name__ == '__main__':
    main()
