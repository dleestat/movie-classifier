import re


def remove_leading_article(string):
    return re.sub("The ", "", string)


def truncate_string(string, max_len):
    if len(string) <= max_len:
        return string
    return f"{string[:max_len - 3]}..."
