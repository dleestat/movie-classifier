import re


def remove_leading_article(string):
    return re.sub("The ", "", string)


def truncate_string(string, max_len):
    return string if len(string) <= max_len else f"{string[:max_len - 3]}..."
