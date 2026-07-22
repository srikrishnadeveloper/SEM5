# Understanding While Loop

## What is a While Loop?

A `while` loop is a control structure that repeatedly executes a block of code **as long as a given condition is `True`**.

Unlike a `for` loop, a `while` loop is mainly used when **you do not know in advance how many times the loop should execute**.

The loop continues until the condition becomes `False`.

---

# Why Do We Use a While Loop?

Suppose you want to keep asking the user for a password until they enter the correct one.

Since you don't know how many attempts the user will take, a `while` loop is the perfect choice.

Example:

```python
password = ""

while password != "python123":
    password = input("Enter Password: ")

print("Access Granted!")
```

The loop continues until the correct password is entered.

---

# Syntax

```python
while condition:
    # Code to execute
```

- **condition** → The loop continues while this is `True`.
- If the condition becomes `False`, the loop stops immediately.

---

# Example 1: Print Numbers from 1 to 5

```python
i = 1

while i <= 5:
    print(i)
    i += 1
```

**Output**

```
1
2
3
4
5
```

---

# How a While Loop Works

1. Initialize a variable.
2. Check the condition.
3. If the condition is `True`, execute the loop body.
4. Update the variable.
5. Go back and check the condition again.
6. Stop when the condition becomes `False`.

---

# Example 2: Finding the Factorial

Factorial of a number `n` is the product of all positive integers from `1` to `n`.

Example:

```
5! = 5 × 4 × 3 × 2 × 1 = 120
```

Python Program:

```python
n = 5
factorial = 1

while n > 0:
    factorial *= n
    n -= 1

print("Factorial of 5 is:", factorial)
```

**Output**

```
Factorial of 5 is: 120
```

---

# Understanding the Factorial Loop

Initial values:

```
n = 5
factorial = 1
```

Iterations:

| Iteration | n | factorial |
|-----------|---|-----------|
| Start | 5 | 1 |
| 1 | 5 | 5 |
| 2 | 4 | 20 |
| 3 | 3 | 60 |
| 4 | 2 | 120 |
| 5 | 1 | 120 |
| End | 0 | Loop Stops |

---

# Infinite Loop

If you forget to update the loop variable, the condition never becomes `False`.

Example:

```python
i = 1

while i <= 5:
    print(i)
```

This loop runs forever because `i` is never increased.

---

# Correct Version

```python
i = 1

while i <= 5:
    print(i)
    i += 1
```

---

# While Loop with User Input

```python
number = int(input("Enter a positive number: "))

while number <= 0:
    number = int(input("Please enter a positive number: "))

print("Valid Input!")
```

The loop keeps asking until the user enters a positive number.

---

# While Loop with `break`

The `break` statement immediately exits the loop.

```python
while True:
    text = input("Type 'exit' to stop: ")

    if text == "exit":
        break

print("Loop Ended")
```

---

# While Loop with `continue`

The `continue` statement skips the current iteration and moves to the next one.

```python
i = 0

while i < 5:
    i += 1

    if i == 3:
        continue

    print(i)
```

**Output**

```
1
2
4
5
```

---

# For Loop vs While Loop

| `for` Loop | `while` Loop |
|------------|--------------|
| Used when the number of iterations is known | Used when the number of iterations is unknown |
| Uses `range()` frequently | Uses a condition |
| Automatically updates the loop variable (through `range()`) | You must update the loop variable manually |
| Less chance of an infinite loop | Higher chance of an infinite loop if not updated |

---

# When Should You Use a While Loop?

Use a `while` loop when:

- The number of iterations is unknown.
- Taking user input until it is valid.
- Reading data until the end.
- Running a program until the user chooses to exit.
- Repeating work until a condition changes.

---

# Summary

- A `while` loop repeats code while a condition is `True`.
- The condition is checked **before** every iteration.
- If the condition is initially `False`, the loop body does **not** execute.
- Always update the loop variable to avoid infinite loops.
- `break` exits the loop immediately.
- `continue` skips the current iteration.
- Use a `while` loop when the number of repetitions is not known beforehand.