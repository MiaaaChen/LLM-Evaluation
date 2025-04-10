import json
from collections import defaultdict

# load the original text dictionary
input_file = "original_text_dict.json"

with open(input_file, "r") as f:
    original_text_dict = json.load(f)

# group the responses by scenario, script, and LLM
grouped_responses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for key, response in original_text_dict.items():
    # get the parts of the key
    parts = key.split("_")

    if len(parts) == 4:
        llm = parts[0]  # "gpt"
        script = parts[1]  # "original"
        scenario = parts[2]  # "s1", "s2"
        index = parts[3]  # "1"

        # get the scenario name
        scenario_name = f"Scenario {int(scenario[1:])}"

        # group the responses
        grouped_responses[scenario_name][script][llm].append(response)

output_file = "responses_by_scenario_script_llm.json"

with open(output_file, "w") as f:
    json.dump(grouped_responses, f, indent=4)

print(
    f"Responses grouped by scenario, script, and LLM have been saved to {output_file}"
)
