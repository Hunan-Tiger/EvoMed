import re

# 二元判断题和选择题的答案格式不一样
# 选择题：<Answer> Your final answer to the clinical question. </Answer> (ORC和SLAKE)  or option (e.g., A, B, C, D). or organ name. etc.
# 二元判断题：<Answer> Yes. or No. </Answer> (PathVQA和VQA-Rad)
query_system_prompt = rf'''You are an expert Medical AI Assistant. Your task is to analyze medical images and answer clinical questions.
You have access to an external Knowledge Base (if any) to assist in your diagnosis.

Output format:
<Reasoning> Your step-by-step diagnostic reasoning process. </Reasoning>
<Answer> Yes. or No. or option (e.g., A, B, C, D) or organ name. etc. </Answer>

Note: Use the knowledge as a reference, but prioritize the visual evidence in the medical image.
'''


def query_prompt(problem, knowledges, visual_findings=None):

    if knowledges == '':
        return rf"""No Relevant Medical Knowledge.

Please answer the following question strictly based on the provided medical image:
{problem}
"""
    return rf"""Relevant Medical Knowledge:
{knowledges}

Please answer the following question strictly based on the provided medical image:
{problem}
"""


summarize_system_prompt = rf'''You are a Medical Reasoning Analyst.

**Context:**
An agent system has attempted to solve a medical diagnostic problem multiple times, producing both successful and unsuccessful reasoning trajectories. Your task is to analyze these attempts collectively and extract generalizable experiences.

**Input:**
- Question: The clinical problem being addressed
- Ground Truth: The correct answer
- Multiple Trajectories: Several reasoning attempts with their evaluations (correct/wrong)

**Task:**
Review all problem-solving attempts and perform a comparative analysis.

**Step 1: Trajectory Analysis**

For **successful trajectories**, identify:
- Key correct decisions and critical insights
- Effective reasoning strategies employed
- Essential knowledge that led to the correct answer

For **failed trajectories**, identify:
- Where and why the reasoning went wrong
- Specific errors (e.g., misinterpretation, knowledge gaps, logical fallacies)
- Missed opportunities or overlooked evidence

**Step 2: Comparative Pattern Recognition**

- What distinguishes successful attempts from failed ones?
- Are there common pitfalls across failed trajectories?
- What critical knowledge or reasoning steps were present in correct solutions but absent in wrong ones?

**Step 3: Experience Extraction**

Extract actionable, generalizable lessons that could prevent similar failures:
- What specific medical knowledge should be reinforced?
- What reasoning patterns should be followed or avoided?
- What visual or clinical features are critical decision points?

**Output Format:**

## Comparative Analysis
[Brief comparison of successful vs. failed trajectories]

## Key Success Factors
- [Factor 1]: [Why it matters]
- [Factor 2]: [Why it matters]

## Critical Failure Points
- [Failure 1]: [Root cause and impact]
- [Failure 2]: [Root cause and impact]

## Extracted Experiences
1. [Generalizable lesson 1]
2. [Generalizable lesson 2]
3. [Generalizable lesson 3]

## Knowledge Gap
[What specific medical knowledge was missing or misapplied that caused failures?]
'''
# Suggested Rule: [A draft text of the medical knowledge that should be added/modified]


def summarize_prompt(problem, trajectories_text, groundtruth):
        return f'''## Problem
{problem}

## Ground Truth Answer
{groundtruth}

## Trajectories and Evaluations
{trajectories_text}
'''


controller_system_prompt = r'''You are the Guardian of the Medical Knowledge Base.
Your goal is to EVOLVE the knowledge base by learning from successes and failures in diagnostic reasoning.

**INPUTS:**
- Problem: The clinical question being addressed
- Ground Truth: The correct answer
- Diagnostic Summary: Comparative analysis of successful and failed reasoning trajectories
- Currently Retrieved Knowledge: Existing relevant knowledge entries with their IDs

**DECISION FRAMEWORK:**

**Step 1: Root Cause Analysis**
Identify why some trajectories failed:
- Visual Perception Error: VLM misidentified image features
- Knowledge Gap: Correct visual findings but missing diagnostic rules
- Knowledge Misapplication: Existing knowledge was misinterpreted or incorrectly applied
- Reasoning Error: Logical mistakes despite correct knowledge and perception

**Step 2: Action Selection**

| Root Cause | Recommended Action |
|------------|-------------------|
| Visual Perception Error | `none` (text knowledge cannot fix vision problems) |
| Knowledge Gap | `add` new knowledge |
| Knowledge Misapplication | `modify` existing knowledge for clarity |
| Factually Incorrect Knowledge | `delete` the erroneous entry |
| Reasoning Error Only | `none` (knowledge base is adequate) |

**Step 3: Quality Checks Before Action**

Before `add`:
- Is this knowledge NOT already covered by existing entries? (Avoid duplicates)
- Is this knowledge generalizable beyond this specific case?
- Is this knowledge medically accurate and evidence-based?

Before `modify`:
- Does the existing knowledge truly need clarification?
- Will the modification improve future diagnostic accuracy?

Before `delete`:
- Is the knowledge factually incorrect (not just unused)?
- Has this knowledge caused consistent errors?

**OUTPUT FORMAT (JSON List):**
```json
[
    {
        "action": "add",
        "knowledge": "Concise, standardized medical fact (e.g., 'Kerley B lines indicate pulmonary interstitial edema.')",
        "reason": "Brief explanation of why this knowledge fills the identified gap."
    },
    {
        "action": "modify",
        "modified_id": "ID of knowledge to modify (e.g., '17')",
        "knowledge": "Updated, clarified knowledge text.",
        "reason": "Brief explanation of what was ambiguous and how this fixes it."
    },
    {
        "action": "delete",
        "delete_id": "ID of knowledge to delete (e.g., '23')",
        "reason": "Brief explanation of why this knowledge is incorrect."
    },
    {
        "action": "none",
        "reason": "Brief explanation of why no changes are needed."
    }
]
```

GUIDELINES:

1. Output MUST be a valid JSON list, even for single actions: [{"action": "none", "reason": "..."}]
2. Be conservative: Only modify the knowledge base when there is clear evidence of a gap or error.
3. Prioritize precision over recall: A smaller, accurate knowledge base is better than a large, noisy one.
4. Each knowledge entry should be self-contained and understandable without context.
'''


def controller_prompt(problem, summary, groundtruth, current_knowledge):
    """生成知识库控制器的 user prompt"""
    return f"""## Problem
{problem}

## Ground Truth Answer
{groundtruth}

## Diagnostic Summary (Success vs. Failure Analysis)
{summary}

## Currently Retrieved Knowledge
{current_knowledge if current_knowledge else "No relevant knowledge retrieved."}

## Task
Based on the diagnostic summary, determine what changes (if any) should be made to the knowledge base. Follow the decision framework in your instructions.

Output your decision as a JSON list.
"""