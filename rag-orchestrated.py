import os
import base64
import json
import re
from openai import AzureOpenAI
from vectorizer import vectorize_image
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

AZURE_OPENAI_MODEL = os.environ["AZURE_OPENAI_MODEL"]
AZURE_OPENAI_RESOURCE = os.environ["AZURE_OPENAI_RESOURCE"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT=f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"

AZURE_AI_SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
AZURE_AI_SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
AZURE_AI_SEARCH_INDEX_NAME = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]

AZURE_VISION_ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"]
AZURE_VISION_KEY = os.environ["AZURE_VISION_KEY"]

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

# the user can supply a message either as pure text
# or as text with image URL in the format [img=path/to/image.jpg]
# -- example --
# do you have a ball matching this uniform? [img=images/football-uniform-flamingo.jpg]
img_regex = r'\[img=([a-zA-Z0-9_/-]+)\.(jpg|jpeg|png|gif)\]$'

def perform_image_search(text_query, image_vector=None):
    search_client = SearchClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, credential=AzureKeyCredential(AZURE_AI_SEARCH_KEY), index_name=AZURE_AI_SEARCH_INDEX_NAME)

    if image_vector is not None:
        vector_query = VectorizedQuery(vector=image_vector, k_nearest_neighbors=3, fields="image_vector")
        results = search_client.search(  
            search_text=text_query,  
            vector_queries= [vector_query],
            select=["description", "filepath"],
        )  
    else:
        results = search_client.search(  
            search_text=text_query,  
            select=["description", "filepath"],
        )
    
    return results

def construct_user_message(user_input, img_regex):
    img_match = re.search(img_regex, user_input)
    if img_match:
        user_input = re.sub(img_regex, '', user_input)
        image_path = f"{img_match.group(1)}.{img_match.group(2)}"
        try:
            with open(image_path, 'rb') as image_file:
                encoded_image_rag = base64.b64encode(image_file.read()).decode('ascii')

            # vectorize the image and search
            vector_data = vectorize_image(image_path, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY)
            search_results = perform_image_search(user_input, vector_data['vector'])
            
            search_descriptions = []
            for result in search_results:
                search_descriptions.append(result["description"])

            formatted_message = f"{user_input.strip()}\n\ncontext (search results):\n" + "\n".join(search_descriptions)

            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_rag}"}}
                ]
            }
        except FileNotFoundError:
            print(f"File not found: {image_path}")
    else:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input}
            ]
        }

try:
    while True:
        user_input = input("> ")
        print()
        user_message = construct_user_message(user_input, img_regex)
        messages.append(user_message)

        full_response = ''
        stream = azure_openai_client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=messages,
            max_tokens=100,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="")

        if full_response:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": full_response}
                ]
            })

        print()
        print()

except KeyboardInterrupt:
    print("\nChat ended.")