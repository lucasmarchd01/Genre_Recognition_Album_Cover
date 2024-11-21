import base64

from openai import OpenAI

client = OpenAI()

# Define the genres
GENRES = [
    "electronic",
    "rock",
    "folk, world, & country",
    "pop",
    "jazz",
]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def predict_genre(image, genres):
    """
    Predict the music genre based on an album cover image using OpenAI API.

    Args:
        image (str): a base 64 encoded image
        genres (list of str): List of possible genres.

    Returns:
        str: Predicted genre from the model.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"This is an image of an album cover. "
                                f"The possible music genres are: {', '.join(genres)}. "
                                f"Based on the visual style and design, predict the genre. "
                                f"Only predict the genre with no explanation."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {e}"


# Example usage
if __name__ == "__main__":
    # Path to the album cover image
    album_cover_path = "/Users/lucasmarch/Projects/Genre_Recognition_Album_Cover/uploads/Massive_Attack_-_Mezzanine.png"

    img = encode_image(album_cover_path)

    # Predict genre
    predicted_genre = predict_genre(img, GENRES)

    # Display the result
    print(f"Predicted Genre: {predicted_genre}")
