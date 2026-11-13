import json
from typing import Literal
from pydantic import BaseModel
from ollama import chat


class AspectResult(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    evidence: str


class ReviewAnalysis(BaseModel):
    aspects: list[AspectResult]
    overall_sentiment: Literal[
        "positive", "negative", "neutral", "mixed"
    ]


def extract_aspects(review: str):
    response = chat(
        model="qwen3.5:9b",
        think=False,
        messages=[
            {
                "role": "system",
                "content": """
You are a precise aspect-based sentiment analysis system.

Your task is to extract product aspects, sentiment,
and supporting evidence from consumer reviews.

Requirements:

1. Identify only explicitly mentioned product
   attributes or usage-related aspects.

ASPECT NAMING REQUIREMENTS:

- Aspect names must describe product attributes,
  properties, or usage-related characteristics.
- Do not use the product name or object name
  as an aspect.
- Prefer concise and consistent noun phrases.
- For example:
  "The laptop gets hot during gaming"
  should use "operating temperature"
  or "heat generation", not "laptop".
- Keep distinct attributes separate.

2. Do not invent product aspects.
   If a review contains only general praise or
   criticism without a specific attribute,
   return an empty aspects list.

3. Assign one sentiment to each aspect:
   positive, negative, or neutral.

4. EVIDENCE REQUIREMENTS:

   - Evidence MUST be copied verbatim from the
     original review.
   - Each evidence must be ONE continuous substring.
   - Do not combine separate sentences or phrases.
   - Do not use ellipses (...) to connect text.
   - Do not paraphrase or summarize.
   - Preserve the original wording.
   - Prefer the shortest complete phrase that
     clearly supports the sentiment.
   - If no valid continuous evidence exists,
     omit that aspect.

5. Determine overall sentiment based on the
   entire review.

6. Do not provide explanations, reasoning,
   or additional commentary.

Return only structured JSON matching the
provided schema.
                """
            },
            {
                "role": "user",
                "content": f"Analyze this review:\n{review}"
            }
        ],
        format=ReviewAnalysis.model_json_schema(),
        options={
            "temperature": 0,
            "num_ctx": 4096
        }
    )


    content = response.message.content or ""

    if not content.strip():
        print("\n[DEBUG] Model returned empty content.")

        print(
            "Thinking:",
            getattr(response.message, "thinking", None)
        )

        print(
            "Done reason:",
            getattr(response, "done_reason", None)
        )

        print("Full response:", response)

        raise RuntimeError("Ollama returned empty content.")


    try:
        result = ReviewAnalysis.model_validate_json(content)

    except Exception as e:
        print("\n[DEBUG] Invalid JSON response:")
        print(repr(content))
        print("Error:", str(e))
        raise

    return result.model_dump()


# Test

from pathlib import Path
import json

if __name__ == "__main__":
    reviews = [
        "The battery lasts for two days and "
        "the screen is very bright.",
        "The laptop has excellent battery life "
        "and a lightweight design, but its display "
        "brightness is disappointing.",
        "The keyboard feels cheap, the fan is noisy, "
        "and the laptop gets hot during gaming."
    ]

    results = []

    for i, review in enumerate(reviews, start=1):
        print(f"\nProcessing review {i}...")

        try:
            analysis = extract_aspects(review)

            results.append({
                "review_id": f"demo_{i}",
                "review_text": review,
                "analysis": analysis
            })

            print(json.dumps(
                analysis,
                indent=4,
                ensure_ascii=False
            ))

        except Exception as e:
            print(f"Error: {e}")


    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "aspect_extraction_demo.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
