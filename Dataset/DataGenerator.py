import os
import json
import time
import language_tool_python
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI
clientGPT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Anthropic
clientAnthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Grammar checker
tool = language_tool_python.LanguageTool('en-US')

def is_grammatically_correct(text):
    matches = tool.check(text)
    return len(matches) <= 2

def create_translation_prompt(sentence):
    return f"Translate the following Gen Z slang into professional workplace language:\n'{sentence}.' \nProvide a single, clear sentence that would be appropriate in a corporate email or professional setting."

def create_slang_generation_prompt():
    return "Give me a Gen Z slang sentence that would be inappropriate in a corporate email or professional setting. Use modern, casual phrasing. Provide only the slang sentence without any additional context or explanation."

def call_openai(prompt):
    response = clientGPT.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def call_claude(prompt):
    response = clientAnthropic.messages.create(
        model="claude-opus-4-0",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# Data generation pipeline
seen_inputs = set()
seen_outputs = set()
dataset = []

max_entries = 150  # You can increase this later

for i in range(max_entries * 2):  # allow for some bad generations
    print(f"\n🔄 Generating slang sentence {i+1}...")
    try:
        slang_sentence = call_openai(create_slang_generation_prompt())
        cleaned_input = slang_sentence.strip()

        if cleaned_input in seen_inputs:
            print("⚠️ Duplicate input. Skipping.")
            continue

        seen_inputs.add(cleaned_input)

        prompt = create_translation_prompt(cleaned_input)
        if len(dataset) % 2 == 0:
            translated = call_openai(prompt)
            model = "OpenAI"
        else:
            translated = call_claude(prompt)
            model = "Claude"

        cleaned_output = translated.strip()

        if cleaned_output in seen_outputs:
            print("⚠️ Duplicate output. Skipping.")
            continue

        if not is_grammatically_correct(cleaned_output):
            print(f"⚠️ Grammar issue in output: {cleaned_output}")
            continue

        print(f"✅ Pair added:\n• Input: {cleaned_input}\n• Output: {cleaned_output}")
        seen_outputs.add(cleaned_output)

        dataset.append({
            "input": cleaned_input,
            "output": cleaned_output,
            "model": model
        })

        if len(dataset) >= max_entries:
            break

        time.sleep(1.5)  # Avoid hitting API rate limits

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        continue

# Save to JSONL
output_file = "genz_to_corp_dataset.jsonl"
with open(output_file, "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\n")

print(f"\n🎉 Dataset generation complete. Total examples: {len(dataset)}")
print(f"📁 Saved to: {output_file}")