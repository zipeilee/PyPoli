
import sys

class A:
    x: int = 1

print(f"Initial: {A.__annotations__}")
ann = A.__annotations__
ann['y'] = int
print(f"Modified via ref: {A.__annotations__}")

A.__annotations__['z'] = float
print(f"Modified directly: {A.__annotations__}")

del A.__annotations__['x']
print(f"Deleted x: {A.__annotations__}")
