import config as cf

def get_poem_text_by_fn(fn):
    """
    Get poem text by filename.
    Args:
        fn (str): filename
    Returns:
        str: poem text
    """
    with open(fn, "r") as f:
        return f.read().strip()

