import { NextRequest } from "next/server";
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

export const maxDuration = 30;

export async function POST(req: NextRequest) {
  const { messages } = await req.json();

  const result = await streamText({
    model: openai("gpt-4o"),
    messages,
    system: `
You are a helpful assistant specialized in explaining and answering questions about a Python machine learning project.

This project includes:
- Classification of customer opinions using logistic regression and naive Bayes.
- Dimensionality reduction with PCA.
- Clustering using K-means.
- It processes data from a CSV with customer reviews, preprocessed using pandas, nltk, and sklearn.
- Models are implemented in Python and evaluated with metrics like accuracy and confusion matrix.

Your job is to explain how these models work, how the code operates, and help interpret results.
Use simple and concise language, and include Python code examples if relevant.
  `,
  });

  const { textStream } = result;

  return new Response(textStream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
