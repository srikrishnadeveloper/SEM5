# Python Basic Input/Output

## Introduction
When you start learning Python, it's important to focus on the basics first. Input and Output (I/O) are fundamental concepts. Python provides built-in functions like `print()` and `input()` for handling output and input.

---

## No Libraries Required
Unlike C++, Python already has built-in functions for basic input and output.

```python
def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()
```

---

## Output with print()

To print output, we use the `print()` function.

```python
print("Hey, Striver!")
```

**Output:**
```
Hey, Striver!
```

---

## Printing on Multiple Lines

If you use multiple `print()` statements, each prints on a new line automatically.

```python
print("Hey, Striver!")
print("Hey, Striver!")
```

**Output:**
```
Hey, Striver!
Hey, Striver!
```

---

## Printing on the Same Line

Use the `end` parameter.

```python
print("Hey, Striver!", end=" ")
print("Hey, Striver!")
```

**Output:**
```
Hey, Striver! Hey, Striver!
```

---

## Newline Character (`\n`)

You can also use `\n` inside a string.

```python
print("Hey, Striver!\nHey, Striver!")
```

**Output:**
```
Hey, Striver!
Hey, Striver!
```

---

## `\n` vs Multiple `print()`

- `\n` → Inserts a new line inside a string.
- Multiple `print()` statements → Automatically print on separate lines.

```python
print("Hello\nWorld")
```

```python
print("Hello")
print("World")
```

Both produce:

```
Hello
World
```

---

## Taking User Input with `input()`

`input()` is used to take input from the user.

```python
x = input()

print("Value of x:", x)
```

**Input:**
```
10
```

**Output:**
```
Value of x: 10
```

---

## Taking Integer Input

`input()` always returns a string.

Use `int()` to convert it into an integer.

```python
x = int(input())

print("Value of x:", x)
```

**Input:**
```
10
```

**Output:**
```
Value of x: 10
```

---

## Multiple Inputs

Use `split()` to separate values.

```python
x, y = input().split()

print("Value of x:", x, "and y:", y)
```

**Input:**
```
10 20
```

**Output:**
```
Value of x: 10 and y: 20
```

---

## Multiple Integer Inputs

Use `map()` to convert all values into integers.

```python
x, y = map(int, input().split())

print("Value of x:", x, "and y:", y)
```

**Input:**
```
10 20
```

**Output:**
```
Value of x: 10 and y: 20
```

---

## Taking a List as Input

```python
arr = list(map(int, input().split()))

print(arr)
```

**Input:**
```
1 2 3 4 5
```

**Output:**
```
[1, 2, 3, 4, 5]
```

---

## Comments

Single-line comment:

```python
# This is a comment
```

Multi-line comment (commonly written using triple quotes):

```python
"""
This is a
multi-line comment.
"""
```

---

## Complete Example

```python
def main():
    x, y = map(int, input().split())

    print("Value of x:", x)
    print("Value of y:", y)

if __name__ == "__main__":
    main()
```

**Input:**
```
10 20
```

**Output:**
```
Value of x: 10
Value of y: 20
```

---

## Quick Summary

| C++ | Python |
|------|---------|
| `cout` | `print()` |
| `cin` | `input()` |
| `int x;` | `x = int(input())` |
| `cin >> x >> y;` | `x, y = map(int, input().split())` |
| `endl` | `print()` or `\n` |
| `using namespace std;` | Not required |
| `#include<iostream>` | Not required |
| `#include<bits/stdc++.h>` | Not required |