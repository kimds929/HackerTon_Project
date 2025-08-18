from gen_ai import problem_generation_pipeline
import pandas as pd
import time

df = pd.read_csv('./eval_questions.csv')
result_df = pd.DataFrame(columns=['problem_text', 'solution_text', 'answer'])

start = time.time()
for i in range(df.shape[0]):
    given_problem_text = df.loc[i, 'question_text']
    given_problem_difficulty = df.loc[i, 'question_difficulty']
    target_difficulty = 2

    generated_result = problem_generation_pipeline(given_problem_text, given_problem_difficulty, target_difficulty)
    result_df.loc[i] = [generated_result['problem_text'], generated_result['solution_text'], generated_result['answer']]
    print(f'{i}th problem is generated')
end = time.time()

print(f'Elapsed time: {end - start}')
result_df.to_csv('./eval_generated_gpt_o1.csv', index=False)
