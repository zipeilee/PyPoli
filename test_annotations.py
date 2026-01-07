
import sys

class A:
    x: int = 1

print(f"Python version: {sys.version}")
print(f"A.__dict__ keys: {list(A.__dict__.keys())}")
try:
    print(f"A.__annotations__: {A.__annotations__}")
except AttributeError:
    print("A has no __annotations__ attribute")

try:
    print(f"A.__dict__['__annotations__']: {A.__dict__['__annotations__']}")
except KeyError:
    print("A.__dict__ has no '__annotations__' key")

A.__annotations__ = {'y': int}
print("After setting __annotations__...")
print(f"A.__dict__ keys: {list(A.__dict__.keys())}")
try:
    print(f"A.__dict__['__annotations__']: {A.__dict__['__annotations__']}")
except KeyError:
    print("A.__dict__ still has no '__annotations__' key")
