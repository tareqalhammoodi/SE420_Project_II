import pandas as pd
from termcolor import colored
from transformers import AutoTokenizer, AutoModel
import torch
from numpy import dot
from numpy.linalg import norm

# function that calculates cosine similarity.
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# function that converts a list of texts into embeddings using BERT model.
def encode_text(texts, tokenizer, model, device):
    tokens = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# load files from our local path.
print(colored("Loading Excel files...", "light_yellow"))
answer_sheet = pd.read_excel('files/answer_sheet.xlsx')
student_answers = pd.read_excel('files/answers.xlsx')

# parse and preprocess the answer sheet.
print(colored("Parsing and preprocessing the answer sheet...", "light_yellow"))
answer_data = {}
for _, row in answer_sheet.iterrows():
    question_id = row['Question- ID']
    question_text = row['Question']
    expected_answers = {
        1: [ans.strip() for ans in row['SCORE-1'].split(',')],
        2: [ans.strip() for ans in row['SCORE-2'].split(',')],
        3: [ans.strip() for ans in row['SCORE-3'].split(',')],
    }
    answer_data[question_id] = {
        'question_text': question_text,
        'expected_answers': expected_answers
    }

# load BERT model and the tokenizer from https://github.com/stefan-it/turkish-bert.
print(colored("Loading BERT model and tokenizer...", "light_yellow"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased").to(device)
print(colored("Environment setup was done successfully.", "magenta"))
print("--------------------------------------------")

# precompute embeddings for the given expected answers.
print(colored("Precomputing embeddings for expected answers...", "light_yellow"))

for qid, data in answer_data.items():
    question_text = data['question_text']
    print(f"\nProcessing Question ID: {qid}")
    print(f"Question: {question_text}")

    for score, answers in data['expected_answers'].items():
        print(f"  Processing Score: {score}")
        print(f"  Expected Answers: {answers}")
        
        # encode answers and store the embeddings.
        embeddings = encode_text(answers, tokenizer, model, device)
        data['expected_answers'][score] = {
            'answers': answers,
            'embeddings': embeddings
        }
        
        # provide feedback about embeddings.
        print(f"  Embeddings computed for {len(answers)} answer(s) under Score {score}.")

print(colored("\nPrecomputing embeddings for all expected answers was done successfully.", "magenta"))
print("--------------------------------------------")

# score the answers of each student.
print(colored("Scoring student answers...", "light_yellow"))
student_scores = []

for _, row in student_answers.iterrows():
    student_id = row['Student ID']
    print("--------------------------------------------")
    print(colored(f"Scoring answers for Student ID: {student_id}", "cyan"))
    total_score = 0
    question_scores = []

    for qid in ['Q1', 'Q2', 'Q3', 'Q4']:
        student_answer = row[f'{qid}-answer']
        if pd.isna(student_answer):
            print(colored(f"No answer provided for {qid}. Assigning score to 0", "red"))
            question_scores.append({'score': 0, 'similarity': 0, 'matching_answer': None})
            continue

        student_embedding = encode_text([student_answer], tokenizer, model, device)[0]
        best_score, best_similarity, best_answer = 0, 0, None

        for score, answers_data in answer_data[qid]['expected_answers'].items():
            for ans, ans_embedding in zip(answers_data['answers'], answers_data['embeddings']):
                # get the best result among similarties.
                similarity = cosine_similarity(student_embedding, ans_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_score = score
                    best_answer = ans
        
        print(colored(f"Question {qid} has {best_similarity} similarity with the expected answers, Best Score for it is {best_score}", "green"))
        total_score += best_score
        question_scores.append({
            'score': best_score,
            'similarity': best_similarity,
            'matching_answer': best_answer
        })

    print(colored(f"Total Score: {total_score}/12", "light_green" if total_score > 6 else "light_red"))
    student_scores.append({
        'student_id': student_id,
        'total_score': total_score,
        'question_scores': question_scores
    })
print("--------------------------------------------")
print(colored("Scoring completed for all students.", "magenta"))
print("--------------------------------------------")

# generate output file
print(colored("Generating output file...", "light_yellow"))
output_data = []

for student in student_scores:
    row = {
        'Student ID': student['student_id'],
        'Student Total Score': student['total_score']
    }
    for i, qid in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        row[f'{qid}-Predicted Score'] = student['question_scores'][i]['score']
        row[f'{qid}-Matching Answer'] = student['question_scores'][i]['matching_answer']
        row[f'{qid}-Cosine similarity'] = student['question_scores'][i]['similarity']
    output_data.append(row)

output_df = pd.DataFrame(output_data)
output_df.to_excel('files/output.xlsx', index=False)
print(colored("Output file was saved to 'files/output.xlsx' successfully.", "magenta"))
