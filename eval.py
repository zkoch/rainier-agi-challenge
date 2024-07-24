import argparse
import openai
import anthropic
import os
import base64
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_type = "openai"

# Set up Anthropic API key
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

def encode_image_to_data_url(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{encoded_image}"
    return data_url

def get_puzzle_solution(image_path, model):
    image_data_url = encode_image_to_data_url(image_path)
    prompt = "Solve the word puzzle in the image. The answer is a common phrase, idiom, or saying. It is designed to be tricky. Think step by step before attempting to solve. If the conclusion isn't a known phrase, you're probably wrong. Put your thoughts process inside <thinking></thinking> brackets and your final answer in an <answer></answer> bracket."

    if model.startswith("gpt"):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting solution from OpenAI: {e}")
            return None
    elif model.startswith("claude"):
        try:
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data_url.split(",")[1]}}
                        ]
                    }
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error getting solution from Claude: {e}")
            return None

def evaluate_solution(solution, correct_answer, model):
    prompt = f"Is the following solution correct for the puzzle: '{solution}'? The correct answer is: '{correct_answer}'. Answer with ONLY 'yes' or 'no'"

    try:
        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": "You help validate LLM created solutions"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error running evaluation with OpenAI: {e}")

def process_single_puzzle(image_file, text_file, model, image_directory, text_directory):
    image_path = os.path.join(image_directory, image_file)
    text_path = os.path.join(text_directory, text_file)

    if not (os.path.exists(image_path) and os.path.exists(text_path)):
        return None

    with open(text_path, 'r') as file:
        correct_answer = file.read().strip()

    solution = get_puzzle_solution(image_path, model)
    evaluation = evaluate_solution(solution, correct_answer, model)

    return {
        "puzzle_number": int(image_file.split('_')[1].split('.')[0]),  # Assuming filenames are numbered
        "solution": solution,
        "correct_answer": correct_answer,
        "evaluation": evaluation
    }

def main():
    parser = argparse.ArgumentParser(description="RAINIER PUZZLE FOR GREAT AGI")
    parser.add_argument("--model", default="gpt-4o", help="Specify the model to use (e.g., gpt-4-turbo, claude-3-5-sonnet-20240620)")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel evaluations to run")
    args = parser.parse_args()

    # Check if API keys are set
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    if not anthropic_api_key:
        raise ValueError("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")

    image_directory = 'data/images'
    text_directory = 'data/text'
    eval_directory = 'evals'

    if not os.path.exists(eval_directory):
        os.makedirs(eval_directory)

    image_files = sorted(f for f in os.listdir(image_directory) if f.endswith('.jpg') and not f.startswith('.'))
    text_files = sorted(f for f in os.listdir(text_directory) if f.endswith('.txt') and not f.startswith('.'))

    results = []
    yes_count = 0
    no_count = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_puzzle = {executor.submit(process_single_puzzle, img, txt, args.model, image_directory, text_directory): (img, txt)
                            for img, txt in zip(image_files, text_files)}

        for future in as_completed(future_to_puzzle):
            img, txt = future_to_puzzle[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f'Puzzle {result["puzzle_number"]} Evaluation: {result["evaluation"]}')
                    if result["evaluation"].lower() == "yes":
                        yes_count += 1
                    elif result["evaluation"].lower() == "no":
                        no_count += 1
                else:
                    print(f'Puzzle {img.split(".")[0]} files are missing or processing failed.')
            except Exception as exc:
                print(f'Puzzle {img.split(".")[0]} generated an exception: {exc}')

    # Sort results by puzzle number
    results.sort(key=lambda x: x["puzzle_number"])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_filename = f'{eval_directory}/eval_{args.model.replace("-", "_")}_{timestamp}.json'

    with open(eval_filename, 'w') as eval_file:
        json.dump(results, eval_file, indent=4)

    total_evaluations = yes_count + no_count
    success_rate = (yes_count / total_evaluations) * 100 if total_evaluations > 0 else 0

    print('---- Evaluation Summary ----')
    print(f'Model used: {args.model}')
    print(f'Evaluation results saved to {eval_filename}')
    print(f'Total Evaluations: {total_evaluations}')
    print(f'Successful Evaluations (Yes): {yes_count}')
    print(f'Unsuccessful Evaluations (No): {no_count}')
    print(f'Success Rate: {success_rate:.2f}%')

if __name__ == '__main__':
    main()
