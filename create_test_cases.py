import json

test_cases = [
    {
        "job_description": "We are looking for a customer support representative with strong verbal communication skills, empathy, and the ability to manage multiple tickets.",
        "correct_assessments": [
            "Customer Service Simulation",
            "Contact Center Test",
            "Call Center - Data Entry and Communication Skills"
        ]
    },
    {
        "job_description": "Seeking a transcriptionist with fast and accurate typing skills and good command over language and grammar.",
        "correct_assessments": [
            "Transcriptionist Solution",
            "Typing Test - General",
            "Language Skills Assessment"
        ]
    },
    {
        "job_description": "Looking for a warehouse associate to assist in packing, sorting, and maintaining workplace safety protocols.",
        "correct_assessments": [
            "Workplace Safety Solution",
            "Warehouse and Logistics Assessment"
        ]
    },
    {
        "job_description": "Hiring a data analyst proficient in Python, SQL, and data visualization tools. Must have strong analytical and problem-solving abilities.",
        "correct_assessments": [
            "Data Analyst Assessment",
            "IT and Programming Test",
            "Numerical Reasoning Test"
        ]
    },
    {
        "job_description": "We are hiring a bank teller responsible for handling financial transactions, customer interactions, and upselling banking products.",
        "correct_assessments": [
            "Teller with Sales - Short Form",
            "Teller 7.0"
        ]
    }
]

with open("test_cases.json", "w") as f:
    json.dump(test_cases, f, indent=4)

print("✅ test_cases.json has been created.")
