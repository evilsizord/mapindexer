from colorama import Fore, Style



def print_error(message):
    print(f"{Fore.RED}{message}{Style.RESET_ALL}")

def print_time(message):
    print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")



