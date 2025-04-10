import os
import asyncio
import json
import yaml
from pathlib import Path
from tqdm.asyncio import tqdm
from pyrit.prompt_target import AzureOpenAITextChatTarget
from pyrit.score import SelfAskLikertScorer

# Initialize AzureOpenAITextChatTarget
red_teaming_llm = AzureOpenAITextChatTarget(
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
)

# Load responses from responses_by_scenario_script_llm.json
with open("responses_by_scenario_script_llm.json", "r") as f:
    responses = json.load(f)


# Helper function to map scores to the Likert scale (1-5)
def map_score_to_likert(scale, min_score=0, max_score=1):
    """
    Maps a float score to the Likert scale range [1, 5].
    """
    return round(1 + (scale - min_score) * (4 / (max_score - min_score)))


async def score_with_retries(scorer, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            score_list = await scorer.score_text_async(text=text)  # type: ignore
            if not score_list:
                raise ValueError("Scorer returned None or empty response.")
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

        # Load YAML file for Likert scale
        yaml_path = Path("likert_config.yaml")
        with open(yaml_path, "r") as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
            print(f"YAML content loaded: {yaml_content}")

        # Initialize SelfAskLikertScorer
        scorer = SelfAskLikertScorer(
            chat_target=red_teaming_llm,
            likert_scale_path=yaml_path,
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
                        input_text = f"Please evaluate the following response based on the guidelines in the Likert Scale YAML file '{yaml_path.name}'.\n\nResponse:\n{response_text}\n\nMake sure your response follows the required JSON format."

                        print(f"Input to scorer: {input_text}")  # Debug input

                        try:
                            score_list = await score_with_retries(scorer, input_text)
                            print(f"Score list: {score_list}")  # Debug output

                            # Store each score result, including the original response text
                            for score in score_list:
                                # Map raw score to Likert scale
                                raw_score = float(score.score_value)
                                mapped_score = map_score_to_likert(raw_score)

                                scores[scenario][script][model].append(
                                    {
                                        "Text": response_text,
                                        "Value": mapped_score,
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
                        except ValueError as ve:
                            print(f"ValueError: {ve}")
                            scores[scenario][script][model].append(
                                {
                                    "Text": response_text,
                                    "Value": "Error",
                                    "Description": "Scorer returned None or empty response.",
                                    "Rationale": str(ve),
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
        with open("evaluation_likert_results.json", "w") as result_file:
            json.dump(scores, result_file, indent=4)

        print("Scoring results have been saved to evaluation_likert_results.json")

    except Exception as e:
        print("Error occurred:", e)


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
