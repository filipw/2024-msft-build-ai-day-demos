import os
import requests
import json

def vectorize_image(image_path, endpoint, subscription_key):
    api_url = f"{endpoint}/computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"
    
    with open(image_path, 'rb') as file:
        image_data = file.read()

    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": subscription_key
    }

    response = requests.post(api_url, headers=headers, data=image_data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as file:
        return json.load(file)

def main(folder_path, endpoint, subscription_key):
    metadata_path = os.path.join(folder_path, 'metadata.json')
    metadata = load_metadata(metadata_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            
            # vectorize the image
            try:
                vector_data = vectorize_image(image_path, endpoint, subscription_key)
                
                # find the corresponding metadata entry
                for item in metadata:
                    if 'image_blob_path' in item and 'description' in item:
                        if item['image_blob_path'] == filename:
                            # update the vector data with metadata
                            vector_data['content'] = item['description']
                            vector_data['image_blob_path'] = item['image_blob_path']

                vector_file_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.json")
                with open(vector_file_path, 'w') as json_file:
                    json.dump(vector_data, json_file)
                print(f"Vector data saved to {vector_file_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

folder_path = 'process-images/'

if __name__ == "__main__":
    AZURE_VISION_ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"]
    AZURE_VISION_KEY = os.environ["AZURE_VISION_KEY"]
    main(folder_path, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY)