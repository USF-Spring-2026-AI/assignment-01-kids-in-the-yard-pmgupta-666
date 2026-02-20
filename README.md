# CS 562 - Assignment 1  
## LLM Reflection Document  
**Prisha Gupta (Undergraduate)**

---

## Tool Used

I used **ChatGPT (GPT-4)** as a code reviewer after completing my implementation. I did not use it to generate the initial structure of the program. Instead, I used it to evaluate clarity, modularity, and potential improvements.

---

## Prompt Used

My prompt to the LLM (slightly edited for clarity) was:

> I implemented a family tree generator in Python using a `Person` `PersonFactory` and `FamilyTree` class structure. The program reads multiple CSV files for life expectancy birth/marriage rates first names last names and gender probabilities.  
>
> Please review my implementation and suggest improvements in:
> 1. Code structure and modularity  
> 2. Data handling efficiency  
> 3. Random sampling logic  
> 4. Separation of concerns  
> 5. Potential edge cases  
>
> Do not rewrite the entire program just suggest specific improvements and explain why they would help

I refined the prompt slightly to focus on structural feedback rather than a full rewrite.

---

## Differences Between My Implementation and the LLM Suggestions

Overall, most of my implementation remained the same. The LLM did not suggest major changes, but instead focused on refinement.

The main differences were:

- It suggested **adding more comments and clarifying docstrings**, which I incorporated to improve readability.
- It recommended slightly improving loop structures and reducing unnecessary iterations where possible, I did not really do that where it wasnt as required.
- It suggested minor refactoring to make some internal helper logic cleaner and more explicit.
- It recommended optionally adding a random seed for reproducibility.

My original implementation already aligned well with the assignment requirements, so the changes were incremental rather than structural.

---

## Changes I Made Based on LLM Feedback

Based on the suggestions, I:

- Added additional explanatory comments in key sections (especially around sampling logic).
- Slightly reduced redundant iterations in certain loops.
- Clarified docstrings to better describe probabilistic behavior.
- Improved readability in some conditional logic.

These changes improved clarity and maintainability without altering the overall design.

---

## Changes I Refused to Make

I chose not to:

- Over-engineer the program with additional classes or abstraction layers.
- Break the implementation into excessive helper modules beyond the assignment’s scope.
- Introduce complex architectural patterns that would reduce clarity.

overall, I prioritized readability and alignment with the specification over production-level abstraction.

---

## Reflection

Using the LLM functioned more like having a structured code reviewer than a code generator. The core logic and design decisions were my own. The LLM helped refine clarity, improve commenting, and slightly optimize structure, but it did not fundamentally change the implementation.
In this context, the LLM was most useful for polishing rather than designing.