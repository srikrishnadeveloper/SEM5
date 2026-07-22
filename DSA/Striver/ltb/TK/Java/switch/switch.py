# match expression
#     case value1:
#         #code
day = 2

match day:
    case 1:
        print("Monday")

    case 2:
        print("Tuesday")

    case 3:
        print("Wednesday")

    case _: #default
        print("Invalid")

# improtant every case check only value 
# we cannot use the <= and all this just chek exact ones


# Use it when you have one variable and many exact values.

# Examples:

# Days of the week
# Months
# Menu selection
# Grade (A, B, C)
# Commands like "start", "stop", "exit"