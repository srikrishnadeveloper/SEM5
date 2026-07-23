# Functions (Pass by Value and Pass by Reference)

## What are Functions?

A **function** is a reusable block of code that performs a specific task.

Instead of writing the same code multiple times, we write it once inside a function and call it whenever needed.

### Syntax

```python
def function_name(parameters):
    # Code
```

### Example

```python
def greet():
    print("Hello!")

greet()
```

**Output**

```
Hello!
```

---

# Passing Arguments to Functions

Functions can receive values called **arguments**.

```python
def greet(name):
    print("Hello", name)

greet("Krishna")
```

**Output**

```
Hello Krishna
```

---

# What is Pass by Value?

In **Pass by Value**, a **copy** of the data is passed to the function.

Changes made inside the function **do not affect** the original variable.

Think of giving someone a **photocopy** of your resume.

They can write on the photocopy, but your original resume remains unchanged.

---

# Python and Immutable Objects

Python uses **pass-by-object-reference (pass-by-assignment)**.

For **immutable objects** such as:

- `int`
- `float`
- `str`
- `tuple`
- `bool`

it behaves like **pass by value**.

Example:

```python
def modify(a):
    a = a + 10

x = 5

modify(x)

print(x)
```

**Output**

```
5
```

The original value remains unchanged because integers are immutable.

---

# What is Pass by Reference?

In **Pass by Reference**, the function works on the original object.

Changes made inside the function affect the original data.

Think of giving someone your **actual debit card**.

Any transaction affects your real account.

---

# Python and Mutable Objects

For **mutable objects** such as:

- `list`
- `dictionary`
- `set`

Python behaves like **pass by reference**.

Example:

```python
def modify(nums):
    nums.append(10)

numbers = [5]

modify(numbers)

print(numbers)
```

**Output**

```
[5, 10]
```

The original list changes because lists are mutable.

---

# Immutable vs Mutable Objects

## Immutable Objects

Cannot be modified after creation.

Examples:

- `int`
- `float`
- `str`
- `tuple`
- `bool`

Example:

```python
name = "Python"

name = "Java"
```

A new object is created instead of modifying the old one.

---

## Mutable Objects

Can be modified after creation.

Examples:

- `list`
- `dict`
- `set`

Example:

```python
numbers = [10, 20]

numbers.append(30)

print(numbers)
```

**Output**

```
[10, 20, 30]
```

---

# Comparison Table

| Feature | Pass by Value (Immutable) | Pass by Reference (Mutable) |
|----------|---------------------------|------------------------------|
| What is passed? | Reference to an immutable object | Reference to a mutable object |
| Original object modified? | No | Yes |
| Examples | `int`, `float`, `str`, `tuple`, `bool` | `list`, `dict`, `set` |
| Memory Usage | New object created after modification | Same object is modified |

---

# Example 1: Integer

```python
def change(x):
    x += 10

num = 5

change(num)

print(num)
```

**Output**

```
5
```

---

# Example 2: String

```python
def change(text):
    text += " World"

name = "Hello"

change(name)

print(name)
```

**Output**

```
Hello
```

Strings are immutable.

---

# Example 3: List

```python
def change(lst):
    lst.append(100)

arr = [10, 20]

change(arr)

print(arr)
```

**Output**

```
[10, 20, 100]
```

Lists are mutable.

---

# Example 4: Dictionary

```python
def update(student):
    student["age"] = 20

data = {"name": "Krishna"}

update(data)

print(data)
```

**Output**

```
{'name': 'Krishna', 'age': 20}
```

---

# Returning Values from Functions

```python
def square(n):
    return n * n

result = square(5)

print(result)
```

**Output**

```
25
```

---

# Summary

- A **function** is a reusable block of code.
- Values passed to a function are called **arguments**.
- Python uses **pass-by-object-reference (pass-by-assignment)**.
- Immutable objects (`int`, `float`, `str`, `tuple`, `bool`) behave like **pass by value**.
- Mutable objects (`list`, `dict`, `set`) behave like **pass by reference**.
- Changes to immutable objects do not affect the original object.
- Changes to mutable objects affect the original object.
- Functions can return values using the `return` statement.