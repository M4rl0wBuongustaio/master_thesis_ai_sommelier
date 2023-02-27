import concurrent.futures


def count_emojis(args):
    df = args[0]
    emoji_list = args[1]
    return len(df[df.note.str.contains(emoji_list)])


def process(df, emoji_list):
    count = 0
    core_count = 10
    len_df = int(len(df) / core_count)
    df_frames = [df.iloc[i * len_df:(i + 1) * len_df].copy() for i in range(core_count + 1)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [
            executor.submit(count_emojis, [df_frames[i], emoji_list]) for i in range(core_count)]

        for result in concurrent.futures.as_completed(results):
            count = count + result.result()
        return count
