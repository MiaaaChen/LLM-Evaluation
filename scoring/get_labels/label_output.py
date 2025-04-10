import json
import csv

input_file = "evaluation_likert_results.json"
output_file = "label_likert.csv"

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

output_data = []

for scenario_key, scripts in data.items():
    scenario_number = scenario_key.split(" ")[1]
    for script_key, llms in scripts.items():
        for llm_key, responses in llms.items():
            for idx, response in enumerate(responses, start=1):
                # generate index
                index = f"{llm_key}_{script_key}_s{scenario_number}_{idx}"
                # get label
                label = response.get("Value", "Error")
                # add to output_data
                output_data.append({"ID": index, "Likert": label})

# write to csv
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["ID", "Likert"])
    writer.writeheader()
    writer.writerows(output_data)

print(f"Label 文件已生成：{output_file}")
