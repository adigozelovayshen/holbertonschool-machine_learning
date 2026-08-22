#!/usr/bin/env python3
""" Interactive prompt loop script """


def create_loop():
    """
    Prompts the user with 'Q:' and prints 'A:' in response.
    Exits with 'A: Goodbye' if user enters exit, quit, goodbye, or bye.
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        try:
            user_input = input("Q: ").strip()
            if user_input.lower() in exit_words:
                print("A: Goodbye")
                break
            else:
                print("A:")
        except (KeyboardInterrupt, EOFError):
            print("\nA: Goodbye")
            break


if __name__ == '__main__':
    create_loop()
