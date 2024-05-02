from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_on_image(data, image_path):
    grounding = data.get('grounding')
    if not grounding:
        print("No grounding data available.")
        return

    lines = grounding.get('lines')
    if not lines:
        print("No lines data available.")
        return

    spans = lines[0].get('spans')
    if not spans:
        print("No spans data available.")
        return

    # Load the image
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)

    img_width, img_height = image.size

    # Plot each polygon and text
    for span in spans:
        polygon = [(coord['x'] * img_width, coord['y'] * img_height) for coord in span['polygon']]
        poly_path = patches.Polygon(polygon, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(poly_path)
        # Add text annotation at the first point of the polygon
        ax.text(polygon[0][0], polygon[0][1], span['text'], verticalalignment='top', color='white', fontsize=8, backgroundcolor='black')

    plt.show()

if __name__ == "__main__":
    sample_data = {'grounding': {'lines': [{'text': 'The image shows a person dressed in stylish attire, walking confidently. They are wearing a dark teal turtleneck sweater paired with navy blue trousers, which are secured with a red patterned belt. Over the sweater, they have donned an olive green overcoat with a fur-lined collar, adding a touch of luxury to the ensemble. The individual is also carrying a brown leather bag, suggesting they may be on their way to work or an appointment. The background features a brick building with a hint of', 'spans': [{'text': 'a person', 'length': 8, 'offset': 16, 'polygon': [{'x': 0.12349999696016312, 'y': 0.023499999195337296}, {'x': 0.6685000061988831, 'y': 0.023499999195337296}, {'x': 0.6685000061988831, 'y': 0.9975000023841858}, {'x': 0.12349999696016312, 'y': 0.9975000023841858}]}, {'text': 'a dark teal turtleneck sweater', 'length': 30, 'offset': 90, 'polygon': [{'x': 0.2694999873638153, 'y': 0.22550000250339508}, {'x': 0.5115000009536743, 'y': 0.22550000250339508}, {'x': 0.5115000009536743, 'y': 0.7304999828338623}, {'x': 0.2694999873638153, 'y': 0.7304999828338623}]}, {'text': 'navy blue trousers', 'length': 18, 'offset': 133, 'polygon': [{'x': 0.3154999911785126, 'y': 0.6854999661445618}, {'x': 0.5644999742507935, 'y': 0.6854999661445618}, {'x': 0.5644999742507935, 'y': 0.9975000023841858}, {'x': 0.3154999911785126, 'y': 0.9975000023841858}]}, {'text': 'a red patterned belt', 'length': 20, 'offset': 176, 'polygon': [{'x': 0.3375000059604645, 'y': 0.6794999837875366}, {'x': 0.49549999833106995, 'y': 0.6794999837875366}, {'x': 0.49549999833106995, 'y': 0.734499990940094}, {'x': 0.3375000059604645, 'y': 0.734499990940094}]}, {'text': 'an olive green overcoat', 'length': 23, 'offset': 233, 'polygon': [{'x': 0.12449999898672104, 'y': 0.20949998497962952}, {'x': 0.6565000414848328, 'y': 0.20949998497962952}, {'x': 0.6565000414848328, 'y': 0.9975000023841858}, {'x': 0.12449999898672104, 'y': 0.9975000023841858}]}, {'text': 'a fur-lined collar', 'length': 18, 'offset': 262, 'polygon': [{'x': 0.20649999380111694, 'y': 0.20250000059604645}, {'x': 0.5945000052452087, 'y': 0.20250000059604645}, {'x': 0.5945000052452087, 'y': 0.3544999957084656}, {'x': 0.20649999380111694, 'y': 0.3544999957084656}]}, {'text': 'the individual', 'length': 14, 'offset': 324, 'polygon': [{'x': 0.5435000061988831, 'y': 0.8084999918937683}, {'x': 0.6725000143051147, 'y': 0.8084999918937683}, {'x': 0.6725000143051147, 'y': 0.9975000023841858}, {'x': 0.5435000061988831, 'y': 0.9975000023841858}]}]}], 'status': 'Success'}}

    plot_on_image(sample_data, 'images/ManInStreet.jpg')