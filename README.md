## AI-NLP-PROJECT DESCRIPTION

In this project,  the system will assign a final score in range [0 12] to each student. In order to predict the final score for a student follow up the steps below:   
 
1- Build a vector/embedding for each answer given in answer-sheet.xlsx file. Use any version of pretrained BERT based embeddings for Turkish (e.g BERTurk, distiluse-base-multilingual-cased-v1)     
 
Related resources:  
• SentenceTransformers Documentation -> https://sbert.net/,  
• Pretrained BERT implementations for Turkish -> https://github.com/stefan-it/turkish-bert  
• Pretrained BERT implementat&ons for Turkish -> https://sbert.net/docs/sentence_transformer/pretrained_models.html#multilingual-models 
 
(2) For each question, vectorize student answer and compare it with alternative answers in each set, choose the one with highest similarity (use cosine similarity) and assign the regarding score.  
 
The output of the system will be the output.xlsx file that includes StudentID, Student Total Score and question predicted score, matching answer and cosine similarity value for each question.
