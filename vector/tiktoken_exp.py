# just an expermient for tiktoken
# IGNORE THIS FILE
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
print("test")
assert enc.decode(enc.encode("hello world")) == "hello world"
print("test")
print(enc.decode(enc.encode("hello world")) == "hello world")
