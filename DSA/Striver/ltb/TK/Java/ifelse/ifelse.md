# If-Else Statements

## Introduction

Conditional statements are a fundamental concept in programming that allow you to make decisions based on certain conditions.

These statements enable your program to execute different blocks of code depending on whether a condition is **True** or **False**.

Python provides three main conditional statements:

- `if`
- `elif`
- `else`

---

# The `if` Statement

The `if` statement is used to execute a block of code only if a certain condition is **True**.

### Syntax

```python
if condition:
    # Code to execute
```

### Example

```python
age = 20

if age >= 18:
    print("You are an adult.")
```

**Output**

```
You are an adult.
```

---

# The `if-else` Statement

The `else` statement specifies what code should execute if the condition in the `if` statement is **False**.

### Syntax

```python
if condition:
    # Executes if condition is True
else:
    # Executes if condition is False
```

### Example

```python
# Ask the user for their age
age = int(input("Enter your age: "))

# Check if the entered age is greater than or equal to 18
if age >= 18:
    print("You are an adult.")
else:
    print("You are not an adult.")
```

### Input

```
16
```

### Output

```
You are not an adult.
```

### Flow of Control

- If the condition is **True**, the code inside the `if` block executes.
- Otherwise, the code inside the `else` block executes.

---

# The `elif` Statement

Sometimes we need to check multiple conditions.

Instead of writing multiple separate `if` statements, Python provides the `elif` (else if) statement.

### Syntax

```python
if condition1:
    # Block 1
elif condition2:
    # Block 2
elif condition3:
    # Block 3
else:
    # Default block
```

---

# Example: Student Grades

```python
marks = 54

if marks < 25:
    print("Grade: F")

elif marks >= 25 and marks <= 44:
    print("Grade: E")

elif marks >= 45 and marks <= 49:
    print("Grade: D")

elif marks >= 50 and marks <= 59:
    print("Grade: C")

elif marks >= 60 and marks <= 69:
    print("Grade: B")

elif marks >= 70:
    print("Grade: A")

else:
    print("Invalid marks entered.")
```

### Output

```
Grade: C
```

---

# How the `if-elif-else` Ladder Works

Suppose:

```python
marks = 54
```

Python checks each condition from top to bottom.

### Step 1

```python
marks < 25
```

```
54 < 25
```

False ❌

Move to the next condition.

---

### Step 2

```python
marks >= 25 and marks <= 44
```

```
54 >= 25 → True
54 <= 44 → False
```

Overall:

```
True and False = False
```

Move to the next condition.

---

### Step 3

```python
marks >= 45 and marks <= 49
```

```
54 >= 45 → True
54 <= 49 → False
```

Overall:

```
False
```

Move to the next condition.

---

### Step 4

```python
marks >= 50 and marks <= 59
```

```
54 >= 50 → True
54 <= 59 → True
```

Overall:

```
True
```

Python prints

```
Grade: C
```

Then **stops checking the remaining conditions.**

---

# Simplifying the Code

The previous solution works correctly, but it contains unnecessary comparisons.

Instead of checking both the lower and upper limits every time, we can check only the upper limits because Python checks conditions from top to bottom.

### Better Version

```python
marks = 54

if marks < 25:
    print("Grade: F")

elif marks <= 44:
    print("Grade: E")

elif marks <= 49:
    print("Grade: D")

elif marks <= 59:
    print("Grade: C")

elif marks <= 69:
    print("Grade: B")

elif marks <= 100:
    print("Grade: A")

else:
    print("Invalid marks entered.")
```

This version is shorter, easier to read, and easier to maintain.

---

# Comparison Operators

| Operator | Meaning |
|----------|---------|
| `==` | Equal to |
| `!=` | Not equal to |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal to |
| `<=` | Less than or equal to |

---

# Logical Operators

| Operator | Meaning |
|----------|---------|
| `and` | Both conditions must be True |
| `or` | At least one condition must be True |
| `not` | Reverses the result |

### Example

```python
age = 20
citizen = True

if age >= 18 and citizen:
    print("Eligible to vote")
```

---

# Summary

- `if` executes a block when the condition is **True**.
- `else` executes when the condition is **False**.
- `elif` is used to check multiple conditions.
- Python checks conditions from **top to bottom**.
- Once a condition becomes **True**, Python executes that block and skips the rest.
- Use logical operators (`and`, `or`, `not`) to combine conditions.