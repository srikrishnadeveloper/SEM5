# Switch Case Statements

## Introduction

`if-elif-else` statements are flexible and can handle almost any condition.

However, when you need to compare **one variable** against **multiple exact values**, Python (3.10+) provides the `match-case` statement, which works similarly to the `switch` statement in other programming languages.

> **Note:** `match-case` is available only in **Python 3.10 and later**.

---

# Why Use `match-case`?

Use `match-case` when:

- You are comparing **one variable**.
- You have **multiple fixed values** to check.
- The code becomes cleaner than a long `if-elif-else` ladder.

Example:

Instead of:

```python
if day == 1:
    print("Monday")
elif day == 2:
    print("Tuesday")
elif day == 3:
    print("Wednesday")
```

You can write:

```python
match day:
    case 1:
        print("Monday")
    case 2:
        print("Tuesday")
    case 3:
        print("Wednesday")
```

---

# Syntax

```python
match expression:
    case value1:
        # Code

    case value2:
        # Code

    case value3:
        # Code

    case _:
        # Default case
```

- `match` checks the expression.
- `case` checks each possible value.
- `case _` is the **default case**.

---

# Example: Day of the Week

```python
day = int(input("Enter a number (1-7): "))

match day:
    case 1:
        print("Monday")

    case 2:
        print("Tuesday")

    case 3:
        print("Wednesday")

    case 4:
        print("Thursday")

    case 5:
        print("Friday")

    case 6:
        print("Saturday")

    case 7:
        print("Sunday")

    case _:
        print("Invalid")
```

### Input

```
4
```

### Output

```
Thursday
```

---

# How `match-case` Works

Suppose

```python
day = 4
```

Python checks

```
case 1 ❌

case 2 ❌

case 3 ❌

case 4 ✅
```

As soon as it finds a match,

```
Thursday
```

is printed.

Python **does not check the remaining cases.**

---

# The Default Case

The default case executes when no other case matches.

```python
day = 10

match day:
    case 1:
        print("Monday")

    case 2:
        print("Tuesday")

    case _:
        print("Invalid")
```

### Output

```
Invalid
```

`case _` works like the `else` block in an `if-elif-else` statement.

---

# Matching Arithmetic Expressions

The expression after `match` can be the result of a calculation.

```python
x = 10
y = 5

match x + y:
    case 15:
        print("Result is 15.")

    case 20:
        print("Result is 20.")

    case _:
        print("No match found.")
```

### Output

```
Result is 15.
```

---

# Matching Characters

```python
grade = 'B'

match grade:
    case 'A':
        print("Excellent!")

    case 'B':
        print("Good!")

    case _:
        print("Not specified.")
```

### Output

```
Good!
```

---

# Matching Strings

```python
fruit = "apple"

match fruit:
    case "apple":
        print("Apple selected")

    case "banana":
        print("Banana selected")

    case _:
        print("Unknown fruit")
```

### Output

```
Apple selected
```

---

# `if-elif-else` vs `match-case`

## Using `if-elif-else`

```python
day = 3

if day == 1:
    print("Monday")

elif day == 2:
    print("Tuesday")

elif day == 3:
    print("Wednesday")

else:
    print("Invalid")
```

---

## Using `match-case`

```python
day = 3

match day:
    case 1:
        print("Monday")

    case 2:
        print("Tuesday")

    case 3:
        print("Wednesday")

    case _:
        print("Invalid")
```

Both produce the same output.

---

# When Should You Use `match-case`?

✅ Use it when:

- Comparing one variable.
- Comparing against fixed values.
- The values are known beforehand.

Example:

- Days of the week
- Months
- Menu choices
- Grades
- Commands

---

# When Should You Use `if-elif-else`?

Use `if-elif-else` for conditions like:

```python
if age >= 18:
```

```python
if marks > 90:
```

```python
if x % 2 == 0:
```

```python
if a > b:
```

These involve comparisons, ranges, or logical expressions that `match-case` is not designed for.

---

# Key Points

- `match-case` is available in **Python 3.10+**.
- It is Python's equivalent of the `switch` statement.
- `case _` acts as the default case.
- Python stops checking after the **first matching case**.
- `match-case` is best for comparing one variable against multiple exact values.

---

# Summary

| Feature | `if-elif-else` | `match-case` |
|---------|----------------|--------------|
| Multiple conditions | ✅ | ❌ |
| Range checking | ✅ | ❌ |
| Exact value matching | ✅ | ✅ |
| Cleaner for many fixed values | ❌ | ✅ |
| Default block | `else` | `case _` |
| Available | All Python versions | Python 3.10+ |