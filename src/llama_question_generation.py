from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from typing import Optional
from pydantic import BaseModel, Field
import pandas as pd 
import os 
import argparse
from tqdm import tqdm 
import numpy as np



# Define output scheme for the LLM Response
class Question(BaseModel):
    """Directed Question based on Topic and Answer Text"""
    question: str = Field(description="The precise and directed question based on topic and answer")

# Define system message for chat
system_message = (
    "system",
    """
    You are a helpful assistant that generates precise and directed question given a TOPIC_NAME and ANSWER.
    The generated question for the ANSWER should be specifically related to the TOPIC_NAME, make sure that the generated question is not vague. 
    The generated question should be readable by a middle school student, short and strictly related to topic name. Don't hallucinate.
    """
)

# Define prompt for LLM
prompt = ChatPromptTemplate.from_messages(
    [   
        system_message,
        (
            "human", 
            """
            TOPIC_NAME : {topic_name}
            ANSWER : {Answer}
            """
        ),
    ]
)


def create_llm_chain(model = "llama3.1:8b", temperature = 0.8):

    llm = ChatOllama(model=model, temperature = temperature)
    structured_llm = llm.with_structured_output(Question)
    question_chain = prompt | structured_llm
    print("LLM Chain Created")
    
    return question_chain




df = pd.read_excel(r"QA_for_Qgeneration.xlsx")
# df = df.sample(50)
new_df = df.copy()
new_df = new_df[['topic_name', 'Answer']]

new_df = pd.DataFrame({'topic_name':to, 'Answer':an})


generated_responses = []
total = new_df.shape[0]
done = 0 
failed = 0 
with tqdm(total=len(new_df), desc="Processing rows") as pbar:
    for index, row in new_df.iterrows():
        try:
            input_dict = row.to_dict()
            question_chain = create_llm_chain(temperature = 0.2)
            response = question_chain.invoke(input_dict)
            generated_responses.append(response.question)
            done += 1
        except:
            generated_responses.append(None)
            failed += 1

        pbar.update(1)
        pbar.set_postfix(total=total, done=done, failed=failed)

df['new_generated_question'] = generated_responses


print(f">>> Generated question: {generated_responses}")

print(f">> SUCCESS: {done}/{total}")


# df.to_excel(r"questions_generated_03_02.xlsx")