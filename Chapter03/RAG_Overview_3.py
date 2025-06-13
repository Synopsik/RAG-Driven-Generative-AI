from RAG_Index_Based import index_query


def main():
    user_input = "what is a rag store?"
    df, response = index_query(user_input)
    print(df.to_markdown(index=False, numalign="left", stralign="left"))


if __name__ == "__main__":
    main()