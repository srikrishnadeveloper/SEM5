#python stack basic

stack = []
stack.append(10)
stack.append(20)
print(stack)
x = stack.pop()
print(stack[-1])
y = stack.pop()
if not stack:
    print("empty")
print(len(stack))
while stack:
    