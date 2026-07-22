# What are Arrays, Strings?

## What is an Array?

An array is a **linear data structure** used to store multiple values in a single variable.

In Python, we use **lists**, which act as dynamic arrays.

Arrays allow us to **randomly access** elements using their **index**.

In other words, an array is a collection of elements stored in a sequence, where each element can be accessed using its position.

Arrays are useful because accessing an element using its index is very fast.

---

## Let's Visualize Arrays

Suppose we have:

```python
arr = [10, 20, 30, 40, 50]
```

```
Index :   0    1    2    3    4
           ↓    ↓    ↓    ↓    ↓
Value :   10   20   30   40   50
```

To access an element:

```python
print(arr[2])
```

**Output**

```
30
```

---

## Why are Arrays '0' Indexed?

Python follows **0-based indexing**.

The first element starts at index **0**, the second at **1**, and so on.

For an array of length **n**, the valid indices are:

```
0 to n - 1
```

Example:

```python
arr = [10, 20, 30, 40]

print(arr[0])
print(arr[3])
```

**Output**

```
10
40
```

---

## Defining an Array

Unlike C++, Python does not require you to define the datatype or size beforehand.

Simply create a list.

```python
arr = [10, 20, 30, 40]
```

or an empty list:

```python
arr = []
```

or a fixed-size list:

```python
arr = [0] * 5
```

**Output**

```
[0, 0, 0, 0, 0]
```

---

## Can Python Lists Store Different Data Types?

Yes.

Unlike C++ arrays, Python lists can store different data types.

```python
arr = [10, "Python", True, 5.6]
```

However, in DSA and competitive programming, we usually store elements of the **same type**.

Example:

```python
arr = [5, 10, 15, 20]
```

---

## Finding an Element in an Array

There are three common ways.

### 1. Using Index

If you know the index:

```python
arr = [10, 20, 30, 40]

print(arr[2])
```

**Output**

```
30
```

---

### 2. Searching

If you don't know the position:

```python
arr = [10, 20, 30, 40]

print(30 in arr)
```

**Output**

```
True
```

Searching algorithms:

- Linear Search
- Binary Search

---

### 3. Hash-Based Lookup

For repeated searches, use a **set** or **dictionary**.

```python
nums = {10, 20, 30, 40}

print(30 in nums)
```

**Output**

```
True
```

---

# Summary of Arrays

- Arrays (lists) store multiple values in one variable.
- Elements are accessed using **index values**.
- Python uses **0-based indexing**.
- Accessing an element using its index takes **O(1)** time.
- Python lists are **dynamic**.
- Lists can grow or shrink automatically.
- Python lists can store different data types.
- In DSA, we usually use homogeneous lists (same data type).

---

# What are Strings?

A string is a sequence of characters stored in a specific order.

Each character has an index starting from **0**.

Example:

```python
s = "striver"
```

```
Index :   0 1 2 3 4 5 6
Character s t r i v e r
```

Accessing characters:

```python
print(s[0])
print(s[1])
print(s[2])
```

**Output**

```
s
t
r
```

---

# Finding the Length of a String

Use the built-in `len()` function.

```python
s = "striver"

print(len(s))
```

**Output**

```
7
```

---

# Accessing Individual Characters

Strings follow **0-based indexing**.

```python
s = "striver"

print(s[0])
print(s[3])
print(s[6])
```

**Output**

```
s
i
r
```

---

# String Concatenation

Use the `+` operator.

```python
first = "Hello"
second = "World"

print(first + second)
```

**Output**

```
HelloWorld
```

To include a space:

```python
print(first + " " + second)
```

**Output**

```
Hello World
```

---

# Passing, Returning and Assigning Strings

Strings can be assigned just like numbers.

```python
s1 = "Python"

s2 = s1

print(s2)
```

**Output**

```
Python
```

Strings can also be passed to functions.

```python
def greet(name):
    print(name)

greet("Krishna")
```

**Output**

```
Krishna
```

---

# String Immutability

Strings are **immutable**.

This means individual characters cannot be modified.

```python
s = "hello"

# s[0] = "H"
```

This will produce an error.

Instead:

```python
s = "Hello"
```

---

# String Comparison

Use `==` to check whether two strings are equal.

```python
s1 = "apple"
s2 = "apple"

print(s1 == s2)
```

**Output**

```
True
```

Different strings:

```python
s1 = "apple"
s2 = "banana"

print(s1 == s2)
```

**Output**

```
False
```

---

Use `!=` to check whether two strings are different.

```python
s1 = "apple"
s2 = "banana"

print(s1 != s2)
```

**Output**

```
True
```

---

You can also compare characters.

```python
s = "striver"

print(s[0] == 's')
print(s[1] == 'a')
```

**Output**

```
True
False
```