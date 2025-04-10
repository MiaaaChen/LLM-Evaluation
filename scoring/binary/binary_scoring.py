import os
import asyncio
import json, yaml
from pathlib import Path
from tqdm.asyncio import tqdm
from pyrit.prompt_target import AzureOpenAITextChatTarget
from pyrit.score import SelfAskTrueFalseScorer

# Initialize AzureOpenAITextChatTarget
red_teaming_llm = AzureOpenAITextChatTarget(
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
)

# Load responses from responses_by_scenario_script_llm.json
with open("responses_by_scenario_script_llm.json", "r") as f:
    responses = json.load(f)


async def score_with_retries(scorer, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            score_list = await scorer.score_text_async(text=text)  # type: ignore
            return score_list
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                print(
                    f"Invalid JSON response, retrying ({attempt + 1}/{max_retries})..."
                )
                continue
            else:
                raise
        except Exception as e:
            if "Status Code: 500" in str(e) and attempt < max_retries - 1:
                print(f"Server error, retrying ({attempt + 1}/{max_retries})...")
                continue
            else:
                raise e


async def main():
    try:
        scores = {}

        # Calculate total number of responses
        total_responses = sum(
            len(response_list)
            for scripts in responses.values()
            for models in scripts.values()
            for response_list in models.values()
        )

        # Initialize progress bar
        progress_bar = tqdm(
            total=total_responses, desc="Scoring responses", unit="response"
        )

        # Load YAML file for classifier
        yaml_path = Path("binary_config.yaml")
        with open(yaml_path, "r") as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
            print(f"YAML content loaded: {yaml_content}")

        # Initialize SelfAskTrueFalseScorer
        scorer = SelfAskTrueFalseScorer(
            true_false_question_path=yaml_path,
            chat_target=red_teaming_llm,
        )
        # Score each response
        for scenario, scripts in responses.items():
            scores[scenario] = {}
            for script, models in scripts.items():
                scores[scenario][script] = {}
                for model, response_list in models.items():
                    scores[scenario][script][model] = []
                    for response_text in response_list:
                        # Prepare input text
                        input_text = f"Please evaluate the following response based on the guidelines in the YAML file '{yaml_path.name}'.\n\nResponse:\n{response_text}\n\nMake sure your response follows the required JSON format."

                        try:
                            score_list = await score_with_retries(scorer, input_text)
                            # Store each score result, including the original response text
                            for score in score_list:
                                scores[scenario][script][model].append(
                                    {
                                        "Text": response_text,
                                        "Value": score.score_value,
                                        "Description": score.score_value_description,
                                        "Rationale": score.score_rationale,
                                    }
                                )
                        except json.JSONDecodeError:
                            print(f"Invalid JSON response for text: {response_text}")
                            scores[scenario][script][model].append(
                                {
                                    "Text": response_text,
                                    "Value": "Error",
                                    "Description": "Invalid JSON response",
                                    "Rationale": "The response was not in a valid JSON format.",
                                }
                            )
                        except Exception as e:
                            print(f"Error occurred while scoring: {e}")
                            scores[scenario][script][model].append(
                                {
                                    "Text": response_text,
                                    "Value": "Error",
                                    "Description": "Scoring failed",
                                    "Rationale": str(e),
                                }
                            )

                        # Update progress bar
                        progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        # Save scoring results to file
        with open("evaluation_binary_results.json", "w") as result_file:
            json.dump(scores, result_file, indent=4)

        print("Scoring results have been saved to evaluation_binary_results.json")

    except Exception as e:
        print("Error occurred:", e)


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
