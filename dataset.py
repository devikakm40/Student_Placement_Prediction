import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

sl_no = np.arange(1, n + 1)
score_10th = np.round(np.random.uniform(50.0, 98.0, n), 2)
puc_score = np.round(np.random.uniform(50.0, 98.0, n), 2)
puc_streams = np.random.choice(['Science', 'Commerce', 'Arts'], size=n, p=[0.70, 0.20, 0.10])

degree_branch = []
for stream in puc_streams:
    if stream == 'Science':
        branch = np.random.choice(['CSE', 'AIML', 'AIDS', 'ECE', 'EEE', 'Aerospace', 'Mechanical', 'Civil'], 
                                  p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05])
    elif stream == 'Commerce':
        branch = np.random.choice(['B.Com', 'BBA', 'CSE'], p=[0.60, 0.30, 0.10])
    else:
        branch = np.random.choice(['BA', 'BBA', 'B.Com'], p=[0.60, 0.25, 0.15])
    degree_branch.append(branch)

cgpa = np.round(np.random.uniform(5.5, 9.8, n), 2)
internships = np.random.randint(0, 4, n)
projects = np.random.randint(0, 6, n)
aptitude_score = np.random.randint(40, 100, n)
soft_skills_score = np.random.randint(40, 100, n)

role_mapping = {
    'CSE': ['Software Developer', 'Web Developer', 'DevOps Engineer', 'System Analyst'],
    'AIML': ['AI Engineer', 'ML Engineer', 'Data Scientist', 'Software Developer'],
    'AIDS': ['Data Analyst', 'Data Scientist', 'ML Engineer', 'Business Analyst'],
    'ECE': ['Embedded Engineer', 'VLSI Design Engineer', 'Software Developer', 'Telecom Engineer'],
    'EEE': ['Power Systems Engineer', 'Control Systems Engineer', 'Electrical Design Engineer'],
    'Aerospace': ['Aerospace Engineer', 'Avionics Engineer', 'Structural Analyst'],
    'Mechanical': ['Mechanical Design Engineer', 'CAD Engineer', 'Production Engineer'],
    'Civil': ['Structural Engineer', 'Site Engineer', 'CAD Drafter'],
    'B.Com': ['Financial Analyst', 'Accountant', 'Tax Consultant', 'Investment Banker'],
    'BBA': ['Business Analyst', 'HR Executive', 'Marketing Specialist'],
    'BA': ['Content Strategist', 'HR Assistant', 'Public Relations Executive']
}

target_role = [np.random.choice(role_mapping.get(b, ['General Analyst'])) for b in degree_branch]

# --- REFINED PLACEMENT LOGIC ---
# Combined composite metric
composite_score = (
    (score_10th * 0.15) + 
    (puc_score * 0.15) + 
    (cgpa * 10) + 
    (internships * 8) + 
    (projects * 5) + 
    (aptitude_score * 0.30) + 
    (soft_skills_score * 0.20)
)

# Sharp sigmoid threshold for clearer separation (less random noise)
prob = 1 / (1 + np.exp(-(composite_score - 122) / 4.5))

# Apply strict campus hiring filters (CGPA < 6.0 or Aptitude < 50 severely drops chances)
prob = np.where((cgpa < 6.0) | (aptitude_score < 48), prob * 0.1, prob)

placed = (np.random.uniform(0, 1, n) < prob).astype(int)

# Salary calculation
salary_lpa = []
for i in range(n):
    if placed[i] == 0:
        salary_lpa.append(0.0)
    else:
        base = 3.5 + (cgpa[i] - 5.5) * 1.3 + (internships[i] * 0.9) + (projects[i] * 0.5)
        if target_role[i] in ['AI Engineer', 'Data Scientist', 'ML Engineer', 'Investment Banker']:
            base += np.random.uniform(2.0, 4.5)
        salary_lpa.append(round(np.clip(base, 3.5, 28.0), 2))

df = pd.DataFrame({
    'sl_no': sl_no,
    'score_10th': score_10th,
    'puc_score': puc_score,
    'puc_stream': puc_streams,
    'degree_branch': degree_branch,
    'cgpa': cgpa,
    'internships': internships,
    'projects': projects,
    'aptitude_score': aptitude_score,
    'soft_skills_score': soft_skills_score,
    'target_role': target_role,
    'placed': placed,
    'salary_lpa': salary_lpa
})

df.to_csv('students_placement_v2.csv', index=False)
print("Regenerated 'students_placement_v2.csv' with realistic placement signals!")