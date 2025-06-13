def call_llm(client, itext):
    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=0.5,
            system="You are an expert Natural Language Processing exercise expert.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": itext
                        }
                    ]
                }
            ]
        )
        formatted_response = "\n".join([block.text for block in message.content if block.type == "text"])
        return (f"Response:\n"
                f"-----------------\n\n"
                f"{formatted_response}\n\n"
                f"-----------------\n")
    except Exception as e:
        return str(e)

def get_models(client):
    # Call API to get list of current models
    models = client.models.list()
    # Condensed statement to print list of models
    print("\n".join([x.id for x in models]))