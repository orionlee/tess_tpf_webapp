def on_session_created(session_context):
    # to keep a file open for testing `psutil.open_files()`
    print(f"on_session_created() for '{session_context.id}'")
    session_context.dummy_file = open(__file__)
    # advance it a bit, in case just opening a file is some special case
    session_context.dummy_file.readline()


def on_session_destroyed(session_context):
    print(f"on_session_destroyed() for '{session_context.id}'")
    session_context.dummy_file.close()
