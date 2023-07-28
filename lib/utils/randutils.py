import string
import random

def randomString(len):
    return ''.join([random.choice(string.ascii_letters) for _ in range(len)])