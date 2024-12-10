### Context:
We want to learn powerful representations of items stored in tabular data. This means being able to represent numbers, categories, text, images, etc…

To this day, building powerful approaches that can represent both text, numbers, text+numbers and categories have been a challenge.

We want to build encoding schemes that can project any stand-alone number, text, text including numbers or categories into the same latent space.

### Objective:
Design and implement an encoding scheme for numerical data, with flexibility for encoding numbers alone and numbers embedded in textual data. This exercise focuses on your ability to explore new approaches, articulate your thought process, and demonstrate skills in feature engineering and encoding.

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
