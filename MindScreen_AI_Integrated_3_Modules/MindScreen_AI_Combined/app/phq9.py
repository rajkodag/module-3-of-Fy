# PHQ-9 scoring and dataset validation

QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?",
    "Thoughts that you would be better off dead or of hurting yourself in some way?"
]

OPTIONS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}

def get_severity(score):
    if 0 <= score <= 4:
        return "Minimal"
    if 5 <= score <= 9:
        return "Mild"
    if 10 <= score <= 14:
        return "Moderate"
    if 15 <= score <= 19:
        return "Moderately Severe"
    if 20 <= score <= 27:
        return "Severe"
    raise ValueError("Score must be between 0 and 27.")

def calculate_score(answers):
    if len(answers) != 9:
        raise ValueError("Exactly 9 answers are required.")
    if any(a not in OPTIONS for a in answers):
        raise ValueError("Each answer must be 0, 1, 2, or 3.")
    score = sum(answers)
    return score, get_severity(score)

def ask_question(number, question):
    print(f"\nQ{number}. {question}")
    for value, label in OPTIONS.items():
        print(f"  {value} - {label}")

    while True:
        value = input("Your answer (0-3): ").strip()
        try:
            answer = int(value)
            if answer in OPTIONS:
                return answer
        except ValueError:
            pass
        print("Invalid input. Please enter 0, 1, 2, or 3.")
