def truncate_string(string, max_len):
    if len(string) <= max_len:
        return string
    return f"{string[:max_len - 3]}..."
