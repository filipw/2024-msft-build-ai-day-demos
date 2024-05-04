import os
import json
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from image_plotter import plot_on_image, plot_on_image_with_plotly

def bbox_to_polygon(bbox, img_width, img_height):
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    return [
        {'x': x / img_width, 'y': y / img_height},
        {'x': (x + w) / img_width, 'y': y / img_height},
        {'x': (x + w) / img_width, 'y': (y + h) / img_height},
        {'x': x / img_width, 'y': (y + h) / img_height}
    ]

def convert_to_openai_format(dense_captions, img_width, img_height):
    description_text = "The image shows "
    offset = len(description_text)
    spans = []

    for caption in dense_captions:
        text = caption['text']
        description_text += text + ", "
        polygon = bbox_to_polygon(caption.bounding_box, img_width, img_height)
        span = {
            'text': text,
            'length': len(text),
            'offset': offset,
            'polygon': polygon
        }
        spans.append(span)
        offset += len(text) + 2

    description_text = description_text.rstrip(", ")

    json_output = {
        "grounding": {
            "lines": [{
                "text": description_text,
                "spans": spans
            }]
        },
        "status": "Success"
    }
    return json_output

def sample_objects_image_file(image_path):
    try:
        endpoint = os.environ["AZURE_VISION_ENDPOINT"]
        key = os.environ["AZURE_VISION_KEY"]
    except KeyError:
        print("Missing environment variable 'AZURE_VISION_ENDPOINT' or 'AZURE_VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[
            VisualFeatures.OBJECTS,
            VisualFeatures.PEOPLE,
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS
        ]
    )

    return result

def print_image_analysis_results(result):
    print("Image analysis results:")

    if result.caption is not None:
        print(" Caption:")
        print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

    if result.dense_captions is not None:
        print(" Dense Captions:")
        for caption in result.dense_captions.list:
            print(f"   '{caption.text}', {caption.bounding_box}, Confidence: {caption.confidence:.4f}")

    if result.read is not None:
        print(" Read:")
        for line in result.read.blocks[0].lines:
            print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
            for word in line.words:
                print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

    if result.tags is not None:
        print(" Tags:")
        for tag in result.tags.list:
            print(f"   '{tag.name}', Confidence {tag.confidence:.4f}")

    if result.objects is not None:
        print(" Objects:", len(result.objects.list))
        for object in result.objects.list:
            print(f"   '{object.tags[0].name}', {object.bounding_box}, Confidence: {object.tags[0].confidence:.4f}")

    if result.people is not None:
        print(" People:")
        for person in result.people.list:
            print(f"   {person.bounding_box}, Confidence {person.confidence:.4f}")

    if result.smart_crops is not None:
        print(" Smart Cropping:")
        for smart_crop in result.smart_crops.list:
            print(f"   Aspect ratio {smart_crop.aspect_ratio}: Smart crop {smart_crop.bounding_box}")

    print(f" Image height: {result.metadata.height}")
    print(f" Image width: {result.metadata.width}")

if __name__ == "__main__":
    result = sample_objects_image_file("images/filip-conf4.jpg")
    print_image_analysis_results(result)
    oai_result_dict = convert_to_openai_format(result.dense_captions.list, result.metadata.width, result.metadata.height)
    plot_on_image_with_plotly(oai_result_dict, "images/filip-conf4.jpg")
