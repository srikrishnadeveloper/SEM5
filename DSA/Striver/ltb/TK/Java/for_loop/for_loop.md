# Understanding For Loop

## What is a For Loop?

A `for` loop is a control structure that allows you to execute a block of code repeatedly.

It is mainly used when you know **how many times** you want to repeat a task.

Instead of writing the same code multiple times, a `for` loop makes the code shorter, cleaner, and easier to maintain.

---

# Why Do We Use a For Loop?

Suppose you want to print:

```
Hello
Hello
Hello
Hello
Hello
```

Without a loop:

```python
print("Hello")
print("Hello")
print("Hello")
print("Hello")
print("Hello")
```

Using a `for` loop:

```python
for i in range(5):
    print("Hello")
```

Both produce the same output, but the `for` loop is much shorter.

---

# Syntax

```python
for variable in range(start, stop, step):
    # Code to execute
```

### Components

- **variable** → Loop variable (changes every iteration).
- **start** → Starting value (inclusive).
- **stop** → Ending value (exclusive).
- **step** → Increment or decrement value.

---

# Example 1: Print 1 to 10

```python
for i in range(1, 11):
    print("Hey, Striver, this is the", i, "th iteration")
```

**Output**

```
Hey, Striver, this is the 1 th iteration
Hey, Striver, this is the 2 th iteration
Hey, Striver, this is the 3 th iteration
...
Hey, Striver, this is the 10 th iteration
```

---

# Understanding `range()`

## `range(stop)`

Starts from **0** and ends before **stop**.

```python
for i in range(5):
    print(i)
```

**Output**

```
0
1
2
3
4
```

---

## `range(start, stop)`

Starts from **start** and ends before **stop**.

```python
for i in range(1, 6):
    print(i)
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

## `range(start, stop, step)`

Moves by the given **step** value.

```python
for i in range(1, 11, 2):
    print(i)
```

**Output**

```
1
3
5
7
9
```

---

# Flow of a For Loop

1. Initialize the loop variable.
2. Check whether the loop should continue.
3. Execute the loop body.
4. Update the loop variable.
5. Repeat until the condition becomes false.
6. Continue with the next statement after the loop.

---

# Nested For Loops

A `for` loop can contain another `for` loop.

This is called a **nested for loop**.

```python
for i in range(3):
    for j in range(3):
        print("i =", i, ", j =", j)
```

**Output**

```
i = 0 , j = 0
i = 0 , j = 1
i = 0 , j = 2
i = 1 , j = 0
i = 1 , j = 1
i = 1 , j = 2
i = 2 , j = 0
i = 2 , j = 1
i = 2 , j = 2
```

---

# Conditional Statements Inside a For Loop

You can use `if-else` inside a loop.

```python
for i in range(1, 11):
    if i % 2 == 0:
        print(i, "is Even")
    else:
        print(i, "is Odd")
```

**Output**

```
1 is Odd
2 is Even
3 is Odd
4 is Even
...
10 is Even
```

---

# Customizing the Step Value

Increase by 5:

```python
for i in range(1, 26, 5):
    print("i =", i)
```

**Output**

```
i = 1
i = 6
i = 11
i = 16
i = 21
```

---

# Decrementing a For Loop

Use a negative step.

```python
for i in range(10, 0, -1):
    print(i)
```

**Output**

```
10
9
8
7
6
5
4
3
2
1
```

---

# Iterating Over a List

```python
numbers = [10, 20, 30, 40]

for num in numbers:
    print(num)
```

**Output**

```
10
20
30
40
```

---

# Iterating Over a String

```python
name = "Python"

for ch in name:
    print(ch)
```

**Output**

```
P
y
t
h
o
n
```

---

# Summary

- A `for` loop repeats a block of code.
- It is useful when the number of iterations is known.
- `range(stop)` starts from 0.
- `range(start, stop)` starts from `start`.
- `range(start, stop, step)` allows custom increments or decrements.
- Nested `for` loops are useful for matrices and pattern problems.
- `if-else` statements can be used inside a `for` loop.
- A negative `step` creates a reverse loop.
- `for` loops can iterate over lists, strings, tuples, and other iterable objects.