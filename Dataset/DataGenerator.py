# Code to generate a dataset of GenZ slang sentences using OpenAI, and Anthropic.
import openai
import anthropic
'''
This is the prompt used to generate the Gen z slang sentence into a professional workplace language'''
def create_prompt(input_sentence):
    return f"Translate the following Gen Z slang into professional workplace language:\n '{input_sentence}'"




def call_openAI(prompt):
    pass

def call_anthropic(prompt):
    pass