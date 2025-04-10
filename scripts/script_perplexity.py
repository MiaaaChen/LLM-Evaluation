import os
import csv
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("PERPLEXITY_API_KEY")

# Load the Sentence Transformer model
model = SentenceTransformer("all-mpnet-base-v2")


# Function to send prompt to LLM and get response
def get_llm_response(messages):
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    response = client.chat.completions.create(
        model="sonar", max_tokens=1000, messages=messages
    )
    return response.choices[0].message.content


# Function to generate embeddings
def generate_embeddings(text):
    return model.encode([text])[0]


# Function to load prompts from a file
def load_prompts(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


# Function to load jailbreak templates from the folder
def load_templates(folder_path):
    templates = {}
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r") as file:
            templates[filename.split(".")[0]] = file.read()
    return templates


# Function to divide prompts into scenarios based on index
def divide_prompts_into_scenarios(prompts):
    scenarios = {
        "s1": prompts[0:4],
        "s2": prompts[4:8],
        "s3": prompts[8:13],
        "s4": prompts[13:17],
        "s5": prompts[17:22],
        "s6": prompts[22:27],
        "s7": prompts[27:31],
        "s8": prompts[31:36],
        "s9": prompts[36:41],
        "s10": prompts[41:45],
    }
    return scenarios


# Load prompts from the file
prompts = load_prompts("prompts.txt")
templates = load_templates("templates")
scenarios = divide_prompts_into_scenarios(prompts)

# Process each template with each scenario
for template_name, template_content in templates.items():
    output_response_path = (
        f"outputs/perplexity/responses_perplexity_{template_name}.csv"
    )
    output_embedding_path = (
        f"outputs/perplexity/embeddings_perplexity_{template_name}.csv"
    )

    with open(output_response_path, "w", newline="") as csvfile, open(
        output_embedding_path, "w", newline=""
    ) as embfile:
        response_writer = csv.writer(csvfile)
        embedding_writer = csv.writer(embfile)

        # Write headers to the CSV files
        response_writer.writerow(["Index", "Prompt", "Response"])
        embedding_writer.writerow(["Index", "Embedding"])

        # Process each scenario
        for scenario_name, scenario_prompts in scenarios.items():
            # Initialize messages for each scenario
            messages = []
            for idx, prompt in enumerate(scenario_prompts, start=1):

                full_prompt = f"{prompt} {template_content}"
                # Append the user's message in the correct format
                messages.append({"role": "user", "content": full_prompt})

                # Get the response for the current round of conversation
                response_text = get_llm_response(messages).replace("\n", " ").strip()

                # Append the assistant's response to messages for continuity
                messages.append({"role": "assistant", "content": response_text})

                # Generate embeddings for the current response
                embedding = generate_embeddings(response_text)

                # Combine index for easier reference, e.g., 's1_0'
                combined_index = f"{scenario_name}_{idx}"

                # Write to CSV files
                response_writer.writerow([combined_index, full_prompt, response_text])
                embedding_str = ",".join(map(str, embedding))
                embedding_writer.writerow([combined_index, embedding_str])

                print(f"Processed {template_name} for prompt index: {combined_index}")

print("Completed processing and saved responses and embeddings.")
