from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

def build_prompt(question, retrieved_chunks):
    prompt = "Answer strictly based on the context below. If not found, say 'I don't know'.\n\n"
    for i, chunk in enumerate(retrieved_chunks):
        prompt += f"Context {i+1}: {chunk}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

def ask_watsonx(prompt, api_key, url, project_id, model_id='mistralai/mistral-large'):
    client = Model(
        model_id=model_id,
        credentials={'apikey': api_key, 'url': url},
        project_id=project_id
    )

    params = {
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.5
    }

    return client.generate_text(prompt=prompt, params=params)
