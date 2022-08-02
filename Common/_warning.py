import warnings

_check_and_warn_enabled = True


def check_and_warn(check_result: bool, message: str):
    if _check_and_warn_enabled and check_result:
        warnings.warn(message, category=RuntimeWarning)
    return


def checked_and_warn(message: str):
    if _check_and_warn_enabled:
        warnings.warn(message, category=RuntimeWarning)
    return


def set_check_and_warn_enabled(value: bool = True):
    global _check_and_warn_enabled
    _check_and_warn_enabled = value
    return _check_and_warn_enabled


def is_check_and_warn_enabled(value: bool = True):
    global _check_and_warn_enabled
    return _check_and_warn_enabled


if __name__ == "__main__":
    check_and_warn(True, "Hello warnings! (default)")
    set_check_and_warn_enabled(False)
    check_and_warn(True, "Hello warnings! (_check_and_warn_enabled = False)")
    set_check_and_warn_enabled(True)
    check_and_warn(True, "Hello warnings! (_check_and_warn_enabled = True)")
