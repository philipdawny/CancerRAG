from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from typing import Optional
from pydantic import BaseModel, Field
import pandas as pd 
import os 
import argparse
from tqdm import tqdm 

# Define output scheme for the LLM Response
class Score(BaseModel):
    """Score of the Question and Answer pairs evaluation"""
    score: int
    reasoning: str




system_message = (
    "system",
    """
    You are a helpful judge who generates precise score and reasoning of question-answer pairs given the ANSWER and the GENERATED_QUESTION.
    Adhere strictly to the provided scale which helps capture both the semantic alignment of the question to the content and the practicality of answering it:

    - 1 = Completely Irrelevant: The GENERATED_QUESTION is entirely unrelated to the ANSWER, or it cannot be answered using the provided information.
    - 2 = Mostly Irrelevant: The GENERATED_QUESTION has minor connections to the ANSWER but is largely unrelated or confusing, making it difficult to answer accurately.
    - 3 = Partially Relevant: The GENERATED_QUESTION is somewhat related to the ANSWER but may lack clarity, specificity, or full alignment, making it only partially answerable.
    - 4 = Mostly Relevant: The GENERATED_QUESTION is closely tied to the ANSWER, is generally clear, and can be answered with reasonable confidence, though it might have minor issues (e.g., slightly ambiguous wording).
    - 5 = Completely Relevant: The GENERATED_QUESTION is fully aligned with the ANSWER, is clear and precise, and can be confidently and completely answered using the available information.
     

    Make sure that your reasoning agrees with the score that you assign to the question answer pair.

    Return the output strictly in the JSON format that includes fields for:
        "score",
        "reasoning"
    """
)

base_url = "http://127.0.0.1:11434/"

# Define prompt for LLM
prompt = ChatPromptTemplate.from_messages(
    [   
        system_message,
        (
            "human", 
            """
            GENERATED_QUESTION : {new_generated_question} 
            ANSWER : {Answer} 
            """
        ),
    ]
)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input-data-path", type=str, help="local path of input dataset")
    # parser.add_argument("--output-path", type=str, help="Path to save output file")

    # args = parser.parse_args()

    # input_data_path = args.input_data_path
    # output_path = args.output_path

    # assert input_data_path is not None and os.path.exists(input_data_path), f"{input_data_path} does not exists"
    # assert output_path is not None, "output path can't be None"

    # Create parent directories
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # print("Input Dataset Path:", input_data_path)
    # print("Output Path:",output_path)
    
    llm = ChatOllama(model="llama3.1:70b", base_url = base_url)
    output_parser = PydanticOutputParser(pydantic_object=Score)
    question_chain = prompt | llm | output_parser
    print("LLM Chain Created")


    input_data_path = r"/nfs/home/tgv3756/CancerRAG/questions_generated_03_02.xlsx"

    df = pd.read_excel(input_data_path)

    print(f">> DF OCLUMNSL : {df.columns}")

    # filter out the rows containing None
    # df = df.dropna()

    # df = df.sample(15)

    generated_score = []
    generated_reasoning = []

    total = df.shape[0]
    done = 0 
    failed = 0 
    with tqdm(total=len(df), desc="Processing rows") as pbar:
        for index, row in df.iterrows():
            try:
                input_dict =  {key: row[key] for key in ['new_generated_question', 'Answer']}
                response = question_chain.invoke(input_dict)
                generated_score.append(response.score)
                generated_reasoning.append(response.reasoning)
                done += 1
            except:
                generated_score.append(None)
                generated_reasoning.append(None)
                failed += 1
            
            pbar.update(1)
            pbar.set_postfix(total=total, done=done, failed=failed)

    df['new_generated_score'] = generated_score
    df['new_generated_reasoning'] = generated_reasoning

    
    
    
    df.to_csv(r"llm_eval_02_07_1036.csv")


