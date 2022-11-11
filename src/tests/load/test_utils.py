import random
import string


alphabet = string.ascii_letters + string.digits


def generate_query():
    size = random.randint(0, 50)
    return ''.join(random.choices(alphabet, k=size))
