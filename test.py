from dotenv import load_dotenv
import os

load_dotenv()

print("RAW:", repr(os.getenv("API_KEY")))
with open(".env", "rb") as f:
    print(f.read())

number = 10


def test():
    print("the number =======: ", number)



test()
