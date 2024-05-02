import os
import base64
import json
import re
import sys
from openai import AzureOpenAI

AZURE_OPENAI_MODEL = os.environ["AZURE_OPENAI_MODEL"]
AZURE_OPENAI_RESOURCE = os.environ["AZURE_OPENAI_RESOURCE"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT=f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"

AZURE_VISION_ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"]
AZURE_VISION_KEY = os.environ["AZURE_VISION_KEY"]
AZURE_VISION_INDEX = os.environ["AZURE_VISION_INDEX"]
VIDEO_OVERRIDE = os.environ.get("VIDEO_OVERRIDE", None)

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
          "text": "You are a video security officer analyzing camera feeds looking for anomalies. Don't talk about keyframes or frames or reference any timestamps explicitly in your description."
        }
      ]
    },
]

extra_body = {
  "enhancements": {
        "video": {
          "enabled": True
        }
    },
  "data_sources": [
    {
      "type": "azure_computer_vision_video_index",
        "parameters": {
            "computer_vision_base_url": f"{AZURE_VISION_ENDPOINT}computervision",
            "computer_vision_api_key": AZURE_VISION_KEY,
            "index_name": AZURE_VISION_INDEX,
            "video_urls": [f"{VIDEO_OVERRIDE}"]
        },
    }
  ],
}

# the user can supply a message either as pure text
# or as text with video URL in the format [video=path/to/video.mp4]
# -- example --
# You are a video security officer analyzing camera feeds looking for anomalies. First provide a summary of whether there is an anomaly occurring that should be looked into further. Second describe the video in detail paying close attention to what is going on in the video. [video=FireDamage.mp4]
img_regex = r'\[video=([a-zA-Z0-9_/-]+)\.(mp4)\]$'

try:
    while True:
        user_input = input("> ")
        print()
        img_match = re.search(img_regex, user_input)
        image_path = ''
        if img_match:
            user_input = re.sub(img_regex, '', user_input)
            video_path = f"{img_match.group(1)}.{img_match.group(2)}"
            try:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input.strip()},
                        {"type": "acv_document_id", "acv_document_id": "fire-damage-1"}
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
            max_tokens=200,
            extra_body=extra_body,
            stream=True,
            top_p=0,
            temperature=0
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if isinstance(chunk.choices[0].delta.content, str):
                    print(chunk.choices[0].delta.content, end="")
                    full_response += chunk.choices[0].delta.content
            
            if chunk.choices and hasattr(chunk.choices[0], 'messages'):
                for message in chunk.choices[0].messages:
                    if 'delta' in message and 'content' in message['delta']:
                        print(" --> TOOL: ", message['delta']['content'])
                        print()

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