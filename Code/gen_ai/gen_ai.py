from openai import OpenAI
import numpy as np
import json
import re

llm_model = "YOUR_MODEL_NAME"

llm = OpenAI(
    api_key="YOUR_API_KEY"
)

def generate_template_from_problem(problem_text, session, problem_difficulty_text, target_difficulty_text):
    session.append({
        "role":"user",
        "content": f"""
        Extract a generalized template from the given math problem. Ensure that:
        - The template maintains the original structure of the problem but removes specific numerical values or unique identifiers, replacing them with appropriate placeholders (e.g., "a", "b", "c").
        - The mathematical logic and format are preserved so that the template can be parameterized later.
        - LaTeX formatting is used for mathematical expressions to ensure correct representation.
        - The difficulty of the given problem is {problem_difficulty_text}. Target difficulty of the generated problem template is {target_difficulty_text}.
        Your response must be in Korean and should not include any additional explanations or formatting markers such as ``` or `template`.

        Here is the given problem:
        {problem_text}

        The final response should be the generalized template in plain text, ready for parameterization, without any additional explanations.
        """
    })
    response = llm.chat.completions.create(
        model=llm_model,
        messages=session
    )
    session.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    return response.choices[0].message.content, session

def parameterize_template(template, session):
    session.append({
        "role":"user",
        "content": f"""
        Using the provided template, create a new math problem by inserting appropriate parameters.
        Ensure the new problem differs from the original by using different numerical values.
        The problem must be a multiple-choice question with exactly five options, where only one is correct.
        Represent the options as simple numbers (1, 2, 3, 4, 5), even if the original problem uses symbols like ①②③④⑤.
        Use integer parameters that are mathematically valid and suitable for middle school students to understand.
        You may use LaTeX code for mathematical expressions, but be precise to ensure correct rendering.
        Use a single backslash for LaTeX commands, as double backslashes should not be used.

        Generating an incorrect problem may negatively impact students, so carefully review and ensure correctness before finalizing your answer.

        Your response must be in Korean. Ensure that all problem content, options, and the answer are presented in Korean.

        Here is the provided template:
        {template}

        Your final output should be formatted as follows:
        - Start with the problem statement and choices(five options ranging from 1 to 5).
        - Follow with "SOLUTION:" and the detailed solution.
        - End with "ANSWER:" followed by the correct answer as a single digit.

        Do not include any additional explanations or formatting markers such as ``` or `json`.
        """
    })
    response = llm.chat.completions.create(
        model=llm_model,
        messages=session
    )
    session.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    return response.choices[0].message.content, session

def regenerate(problem_text, solution_text, answer, session):
    session.append({
        "role":"user",
        "content": f"""
        Review and refine the generated math problem using the solution and answer provided. Ensure that:
        - The problem includes the correct answer within the given options.
        - If necessary, modify the choices to include the correct answer and adjust the options to maintain problem validity.

        Here is the initial problem text:
        {problem_text}

        Here is the solution:
        {solution_text}

        Your initially proposed answer was: {answer}. Feel free to adjust the answer if needed.
        Key is to ensure that the problem, solution, and answer are consistent and mathematically accurate.
        You should modify the choices to include the correct answer found in the solution and provide that as the final answer.

        Modify and regenerate the problem if needed. Your output should include:
        - Start with the problem statement and revised choices(five options ranging from 1 to 5, which must include the answer).
        - Follow with "SOLUTION:" and the detailed solution.
        - End with "ANSWER:" followed by the correct answer as a single digit.

        Do not include any additional explanations or formatting markers like ``` or `json`.
        """
    })
    response = llm.chat.completions.create(
        model=llm_model,
        messages=session
    )
    session.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    return response.choices[0].message.content, session


def problem_generation_pipeline(given_problem_text, problem_difficulty, target_difficulty):
    """
    @param given_problem_text: The original problem text in Korean.
    @param problem_difficulty: The difficulty level of the original problem (1: Easy, 2: Medium, 3: Hard).
    @param target_difficulty: The target difficulty level for the generated problem (1: Easy, 2: Medium, 3: Hard).
    @return: A dictionary containing the generated problem text, solution text, and answer.
    """
    try:
        session = [
            {
                "role": "system",
                "content": """
                You are an expert in mathematics education, specialized in creating and solving math problems for K-12 students in Korea.
                Although your native language is English, you are fluent in Korean and capable of writing problems and detailed solutions in Korean.
                Your expertise includes:
                - Creating mathematically accurate and educationally appropriate problems.
                - Solving problems with clear, concise explanations tailored to middle and high school students.
                - Using LaTeX for formatting mathematical expressions where necessary to ensure clarity.

                Always provide responses in Korean, ensuring that the language and difficulty level are suitable for K-12 students.
                When writing problems or solutions, focus on clarity, step-by-step reasoning, and educational value.
                """
            }
        ]

        # target_difficulty 3 -> Hard, 2 -> Medium, 1 -> Easy
        target_difficulty_text = ["Easy", "Medium", "Hard"][target_difficulty - 1]
        problem_difficulty_text = ["Easy", "Medium", "Hard"][problem_difficulty - 1]

        generated_template, session = generate_template_from_problem(given_problem_text, session, problem_difficulty_text, target_difficulty_text)
        generated_problem_solution_answer, session = parameterize_template(generated_template, session)

        problem_text, rest = generated_problem_solution_answer.split("SOLUTION:", 1)
        solution_text, answer = rest.rsplit("ANSWER:", 1)
        answer = answer.strip().strip('.')

        regenerated_problem_solution_answer, session = regenerate(problem_text, solution_text, answer, session)
        regenerated_problem_text, rest = regenerated_problem_solution_answer.split("SOLUTION:", 1)
        regenerated_solution_text, regenerated_answer = rest.rsplit("ANSWER:", 1)
        regenerated_answer = regenerated_answer.strip().strip('.')
        if "," in regenerated_answer:
            regenerated_answer = regenerated_answer.split(",")[0]

        response = {'problem_text' : regenerated_problem_text.strip(), 'solution_text' : regenerated_solution_text.strip(), 'answer' : regenerated_answer.strip()}

        return response
    except Exception as e:
        pass



#def verify_with_sympy(generated_problem_text, session):
#     session.append({
#         "role":"user",
#         "content": f"""
#         Write SymPy code to verify if the generated problem can be solved using SymPy. Ensure the code checks all answer choices using a loop to verify them one by one.
#         Execute the SymPy code on your own and return the correct answer as a single digit number between 1 and 5.

#         Your response should be in Korean and should only include the answer as a single number without any explanations or additional text.
#         The number must be within the range of 1 to 5, which can be used for direct comparison with the previously provided answer.
#         """
#     })
#     response = llm.chat.completions.create(
#         model=llm_model,
#         messages=session
#     )
#     session.append({
#         "role": "assistant",
#         "content": response.choices[0].message.content
#     })
#     return response.choices[0].message.content, session

#def generate_solution(answer_pair_equal, session):
#     if answer_pair_equal:
#         session.append({
#             "role": "user",
#             "content": f"""
#             The answer generated by SymPy matches the answer provided by the user, confirming that the problem is mathematically valid.
#             Write a detailed, step-by-step solution for the generated problem. The solution should:
#             - Clearly explain the logic and reasoning behind the correct answer.
#             - Be concise and easy to understand, suitable for middle school students.
#             - Include all necessary steps, detailed explanations, and mathematical justifications to guide students through the solution process.

#             Make sure your response is written in Korean and does not include any additional formatting markers or explanations, such as ``` or `solution`.
#             """
#         })
#     else:
#         session.append({
#             "role": "user",
#             "content": f"""
#             The answer generated by SymPy does not match the answer provided by the user, indicating a potential issue with the problem.
#             You might need to review the problem and choose the right answer based on the correct mathematical reasoning.
#             Write a detailed, step-by-step solution for the generated problem, highlighting any discrepancies or issues that may have led to the mismatch.
#             The solution should:
#             - Clearly explain the logic and reasoning behind the correct answer.
#             - Be concise and easy to understand, suitable for middle school students.
#             - Include all necessary steps, detailed explanations, and mathematical justifications to guide students through the solution process.

#             Make sure your response is written in Korean and does not include any additional formatting markers or explanations, such as ``` or `solution`.
#             """
#         })
#     response = llm.chat.completions.create(
#         model=llm_model,
#         messages=session
#     )
#     session.append({
#         "role": "assistant",
#         "content": response.choices[0].message.content
#     })
#     return response.choices[0].message.content, session
