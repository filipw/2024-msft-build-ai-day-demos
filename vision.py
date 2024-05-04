import os
import base64
import json
import re
import sys
from openai import AzureOpenAI
from image_plotter import plot_on_image, plot_on_image_with_plotly

AZURE_OPENAI_MODEL = os.environ["AZURE_OPENAI_MODEL"]
AZURE_OPENAI_RESOURCE = os.environ["AZURE_OPENAI_RESOURCE"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT=f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"

AZURE_VISION_ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"]
AZURE_VISION_KEY = os.environ["AZURE_VISION_KEY"]

azure_openai_client = AzureOpenAI(
            api_version="2024-02-15-preview",
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

messages = []

extra_body = {
    "enhancements": {
        "ocr": {
          "enabled": True
        },
        "grounding": {
          "enabled": True
        }
    },
  "data_sources": [
    {
      "type": "AzureComputerVision",
        "parameters": {
            "endpoint": AZURE_VISION_ENDPOINT,
            "key": AZURE_VISION_KEY
        }
    }
  ],
}

# the user can supply a message either as pure text
# or as text with image URL in the format [img=path/to/image.jpg]
# -- example --
# describe the image [img=images/filip-conf4.jpg]
img_regex = r'\[img=([a-zA-Z0-9_/-]+)\.(jpg|jpeg|png|gif)\]$'

try:
    while True:
        user_input = input("> ")
        print()
        img_match = re.search(img_regex, user_input)
        image_path = ''
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
            max_tokens=800,
            extra_body=extra_body,
            stream=True,
            top_p=0,
            temperature=0
        )

        image_data_to_plot = ''
        for chunk in stream:
          if chunk.choices and chunk.choices[0].delta.content:
              if isinstance(chunk.choices[0].delta.content, str):
                  print(chunk.choices[0].delta.content, end="")
                  full_response += chunk.choices[0].delta.content
                  sys.stdout.flush() # flush the buffer buffer explicitly
              else:
                  print("  [see browser window for more details]   ")
                  image_data_to_plot = chunk.choices[0].delta.content

        if full_response:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": full_response}
                ]
            })

        if image_data_to_plot != '' and image_path != '':
            plot_on_image_with_plotly(image_data_to_plot, image_path)

        img_match = None
        print()
        print()

except KeyboardInterrupt:
    print("\nChat ended.")