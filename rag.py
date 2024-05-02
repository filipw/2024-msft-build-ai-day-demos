import os
import base64
import json
import re
from openai import AzureOpenAI

AZURE_OPENAI_MODEL = os.environ["AZURE_OPENAI_MODEL"]
AZURE_OPENAI_RESOURCE = os.environ["AZURE_OPENAI_RESOURCE"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT=f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"

AZURE_AI_SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
AZURE_AI_SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
AZURE_AI_SEARCH_INDEX_NAME = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]

azure_openai_client = AzureOpenAI(
            api_version="2024-02-15-preview",
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

messages = [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are sports store assistant. you only speak about the products available to you in the context."
        }
      ]
    },
]

extra_body = {
  "data_sources": [
    {
      "type": "azure_search",
      "parameters": {
        "endpoint": AZURE_AI_SEARCH_ENDPOINT,
        "authentication": {
            "key": AZURE_AI_SEARCH_KEY,
            "type": "api_key"
        },
        "key": AZURE_AI_SEARCH_KEY,
        "index_name": AZURE_AI_SEARCH_INDEX_NAME
      }
    }
  ],
}

# the user can supply a message either as pure text
# or as text with image URL in the format [img=path/to/image.jpg]
# -- example --
# do you have a ball matching this uniform? [img=images/football-uniform-flamingo.jpg]
img_regex = r'\[img=([a-zA-Z0-9_/-]+)\.(jpg|jpeg|png|gif)\]$'

try:
    while True:
        user_input = input("> ")
        print()
        img_match = re.search(img_regex, user_input)
        if img_match:
            user_input = re.sub(img_regex, '', user_input)
            image_path = f"{img_match.group(1)}.{img_match.group(2)}"
            try:
                with open(image_path, 'rb') as image_file:
                    encoded_image_rag = base64.b64encode(image_file.read()).decode('ascii')
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input.strip()},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_rag}"}}
                    ]
                })
            except FileNotFoundError:
                print(f"File not found: {image_path}")
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input}
                ]
            })

        full_response = ''
        stream = azure_openai_client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=messages,
            max_tokens=100,
            extra_body=extra_body,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="")

        if full_response:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": full_response}
                ]
            })

        print()
        print()

except KeyboardInterrupt:
    print("\nChat ended.")