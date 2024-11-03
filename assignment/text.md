### Context:
We want to learn powerful representations of items stored in tabular data. This means being able to represent numbers, categories, text, images, etc…

To this day, building powerful approaches that can represent both text, numbers, text+numbers and categories have been a challenge.

We want to build encoding schemes that can project any stand-alone number, text, text including numbers or categories into the same latent space.

### Objective:
Design and implement an encoding scheme for numerical data, with flexibility for encoding numbers alone and numbers embedded in textual data. This exercise focuses on your ability to explore new approaches, articulate your thought process, and demonstrate skills in feature engineering and encoding.

### Instructions:
* Duration: 4 hours (but you can take more time if you want).
* Submit: 
    * The source code and Jupyter notebook or Python scripts used.
    * A brief report (1-2 pages) explaining the approach, rationale behind choices, challenges faced, and potential improvements.
* A README file for code execution and dependencies.

### Problem Statement
Your task is to design a new encoding scheme to represent numerical data, especially when it appears in a structured dataset (e.g., tabular data) or is embedded within text (e.g., "5-star rating" or "23 years old"). The encoding scheme should be flexible enough to capture information on numerical scale, relationships to other numbers, and contextual cues when numbers are presented with text.

### Requirements:
* Design Goals:
    * The encoding scheme should:
        * Capture numerical relationships (e.g., ordinal or interval relationships).
        * Represent numbers alone as well as in the context of text.
        * Preserve semantic meaning in both cases to be usable as embeddings in downstream tasks like classification or clustering.

* Implementation:
    * Implement the encoding scheme in Python, creating a function encode_number(input) where input can  be a number or some text.
    * Test your scheme with the following examples:
        * Standalone numbers
            * number1 = 3.14
            * number2 = 10

        * Numbers with text
            * text1 = "Rated 5 stars"
            * text2 = “The product costs 23 dollars"

### Evaluation:
* Run your encoding scheme on a set of sample data points (you can use synthetic data if necessary).
* Demonstrate how well your scheme preserves numeric relationships (e.g., similar numbers produce similar embeddings).
* For text+number embeddings, show that embeddings vary meaningfully with text context (e.g., "5 stars" vs. "5 units").
* Be creative and rigorous in your evaluation. Design meaningful metrics and explain why each metric is useful for the evaluation of your embedding scheme.

### Report (1-2 pages):
* Approach: Describe your thought process and why you chose your particular encoding scheme.
* Challenges: Outline any issues encountered, such as handling decimal numbers, large integers, or complex text contexts.
* Potential Improvements: Suggest ways to further enhance the scheme, such as incorporating domain-specific information or additional embedding techniques.

### Submission Guidelines:
* Code:
    * Provide clean, well-commented code.
    * Make sure the code is modular and reusable.
    * Include error handling and testing of edge cases (e.g., very large numbers, zero).

* Report:
    * Describe your approach clearly, including your encoding design choices.
    * Identify limitations, if any, and suggest improvements.
    * Demonstrate an understanding of how encoding quality affects downstream ML tasks.

* README:
    * Include clear instructions on running the code.
    * List dependencies and installation steps if applicable.

### Evaluation Criteria:
* Innovation and Creativity:
    * Novelty in the encoding approach and adaptability for different contexts (stand-alone vs. text).
    * Thoughtfulness in addressing the complexities of number-text interactions.
    
* Technical Execution:
    * Correctness and robustness of the implementation.
    * Code quality: clean, well-organized, and efficient.

* Analytical Thinking:
    * Logical approach and well-reasoned explanations in the report.
    * Ability to anticipate limitations and suggest meaningful improvements.
