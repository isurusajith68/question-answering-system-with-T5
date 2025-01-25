def get_mcq_prompt(chunk, question):
    return [
        {
            "role": "user",
            "content": f"""
            {chunk}

            Question: {question}

            Instructions:
            1. Create a multiple-choice question (MCQ) based on the provided content.
            2. The MCQ should have four options:
               - One correct answer.
               - Three plausible but incorrect answers.
               - Shuffle the positions of the correct answer randomly.
            3. Ensure the incorrect options are logically related to the content but not accurate.
            4. The options should be clear, relevant, and contextually appropriate.
            5. No Need Explanation.
            6. Options should be short and clear, 10 to 15 words.
            7. Provide the output in JSON format, like this example:

                {{
                    "question": "{question}",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "answer": correct option index (1, 2, 3, or 4) only (e.g., 1). Do not include any comments in the output.
                }}

            Answer:
            """
        }
    ]